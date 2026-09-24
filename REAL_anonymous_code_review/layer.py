from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_scatter import scatter

from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.conv import MessagePassing, SimpleConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor, Size

from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold

    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class NodeCentricConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            model_weights: tuple = (),
            lambda_tradeoff: float = 0.2,
            cache_source: bool = False,
            uniform_attention_init: bool = False,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda_tradeoff = lambda_tradeoff
        self.cache_source = cache_source
        self._source_cache_key = None
        self._source_cache = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.films = list()
        for weight in model_weights:
            self.films.append(weight.t())

        self.att = Parameter(torch.Tensor(self.out_channels, 1))
        if uniform_attention_init:
            nn.init.zeros_(self.att)
        else:
            nn.init.xavier_normal_(self.att)
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        nn.init.xavier_normal_(self.weight)

        self.neigh_aggr = SimpleConv(aggr='mean')

        self.sparse_attention = Sparsemax(dim=1)

    def _cached_source_terms(self, x: Tensor, edge_index: Adj):
        key = (x.data_ptr(), edge_index.data_ptr(), tuple(x.size()), tuple(edge_index.size()), str(x.device))
        if self._source_cache_key == key and self._source_cache is not None:
            return self._source_cache

        edge_index_loop, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        with torch.no_grad():
            neigh_rep = self.neigh_aggr(x, edge_index_loop).detach()
            base_out = self.propagate(edge_index_loop, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None).detach()
            att_inputs = [torch.matmul(neigh_rep, film).detach() for film in self.films]
            source_reps = [torch.matmul(base_out, film).detach() for film in self.films]

        self._source_cache_key = key
        self._source_cache = (neigh_rep, att_inputs, source_reps)
        return self._source_cache

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_type: OptTensor = None) -> Tensor:
        use_cache = self.cache_source and all(not film.requires_grad for film in self.films)

        if use_cache:
            neigh_rep, att_inputs, reps = self._cached_source_terms(x, edge_index)
            atts = [torch.matmul(rep, self.att) for rep in att_inputs]
        else:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
            neigh_rep = self.neigh_aggr(x, edge_index)

            atts = []
            reps = []

            out = self.propagate(edge_index, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None)

            for i, film in enumerate(self.films):
                rep = torch.matmul(neigh_rep, film)
                res = torch.matmul(rep, self.att)
                atts.append(res)
                rep = torch.matmul(out, film)
                reps.append(rep)

        atts = torch.cat(atts, dim=1)
        w = self.sparse_attention(atts)
        gamma = torch.stack(reps)
        w = w.t().unsqueeze(-1)

        gamma = torch.sum(w * gamma, dim=0)

        if self.lambda_tradeoff == 0:
            out = gamma
        else:
            wg = torch.matmul(neigh_rep, self.weight)
            out = gamma + wg * self.lambda_tradeoff

        return out
    
    def message(self, x_j: Tensor, gamma_i: Tensor, edge_weight: OptTensor) -> Tensor:
        out = gamma_i * x_j
        
        return out


class MLPModule(torch.nn.Module):
    def __init__(self, args, model_list):
        super(MLPModule, self).__init__()
        self.args = args
        self.model_list = model_list

        self.att = Parameter(torch.Tensor(args.num_classes, 1))
        if getattr(args, 'uniform_attention_init', False):
            nn.init.zeros_(self.att)
        else:
            nn.init.xavier_normal_(self.att)

        self.sparse_attention = Sparsemax(dim=1)

    def forward(self, x):
        outputs = []
        weights = []
        for i in range(len(self.model_list)):
            cls_output = self.model_list[i].gnn.cls(x)
            att = torch.matmul(cls_output, self.att)
            outputs.append(cls_output)
            weights.append(att)
        weights = torch.cat(weights, dim=1)
        w = self.sparse_attention(weights)

        outputs = torch.stack(outputs)
        w = w.t().unsqueeze(-1)
        x = torch.sum(w * outputs, dim=0)

        return x
