import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, negative_sampling, to_undirected


EPS = 1e-12


def _assert_shape(name, tensor, expected):
    assert isinstance(tensor, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    assert tensor.dim() == len(expected), '{} shape mismatch: got {}, expected rank {}'.format(
        name, tuple(tensor.shape), len(expected)
    )
    for actual, exp in zip(tensor.shape, expected):
        if exp is not None:
            assert actual == exp, '{} shape mismatch: got {}, expected {}'.format(
                name, tuple(tensor.shape), expected
            )


def _assert_finite(name, tensor):
    if tensor.is_floating_point() or tensor.is_complex():
        assert torch.isfinite(tensor).all(), '{} contains NaN or Inf'.format(name)


def _debug_tensor(name, tensor, debug):
    if not debug:
        return
    shape = list(tensor.shape)
    if tensor.numel() == 0:
        print('[M1] {} shape: {} empty'.format(name, shape))
        return
    stat_tensor = tensor.detach()
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.float()
    print(
        '[M1] {} shape: {} mean={:.6f} min={:.6f} max={:.6f}'.format(
            name,
            shape,
            stat_tensor.mean().item(),
            stat_tensor.min().item(),
            stat_tensor.max().item(),
        )
    )


def _assert_edge_index(name, edge_index, num_nodes=None):
    _assert_shape(name, edge_index, (2, None))
    assert edge_index.dtype == torch.long, '{} must have dtype torch.long'.format(name)
    if edge_index.numel() > 0 and num_nodes is not None:
        assert edge_index.min().item() >= 0, '{} has negative node ids'.format(name)
        assert edge_index.max().item() < num_nodes, '{} has node ids outside num_nodes'.format(name)


def _assert_probability(name, prob, expected_shape):
    _assert_shape(name, prob, expected_shape)
    _assert_finite(name, prob)
    assert (prob >= -1e-6).all(), '{} contains negative probabilities'.format(name)
    row_sum = prob.sum(dim=-1)
    ones = torch.ones_like(row_sum)
    assert torch.allclose(row_sum, ones, atol=1e-4, rtol=1e-4), (
        '{} rows do not sum to 1'.format(name)
    )


@torch.no_grad()
def build_perturbed_edge_views(
    data,
    num_views: int = 5,
    edge_drop: float = 0.10,
    edge_add: float = 0.05,
    force_undirected: bool = True,
    view_policy: str = 'alternate',
):
    """
    Return a list of edge_index views. The first view is the original edge_index.
    """
    assert num_views >= 0, 'num_views must be non-negative'
    assert 0.0 <= edge_drop < 1.0, 'edge_drop must be in [0, 1)'
    assert edge_add >= 0.0, 'edge_add must be non-negative'
    assert view_policy in {'alternate', 'drop_only', 'drop_add'}, 'invalid view_policy'

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    assert num_nodes is not None, 'data.num_nodes must be available'
    _assert_edge_index('edge_index', edge_index, num_nodes)

    views = [edge_index]
    num_edges = edge_index.size(1)

    def _drop_view(base_edge_index):
        new_edge_index, _ = dropout_edge(
            base_edge_index,
            p=edge_drop,
            force_undirected=force_undirected,
            training=True,
        )
        new_edge_index = new_edge_index.to(edge_index.device)
        _assert_edge_index('drop_edge_index', new_edge_index, num_nodes)
        return new_edge_index

    def _add_view(base_edge_index):
        base_edge_index = base_edge_index.to(edge_index.device)
        base_num_edges = base_edge_index.size(1)
        num_add = 0
        if edge_add > 0.0:
            add_base = base_num_edges if base_num_edges > 0 else num_edges
            if add_base > 0:
                num_add = max(1, int(add_base * edge_add))
        if num_add > 0:
            add_edges = negative_sampling(
                edge_index=base_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_add,
            ).to(edge_index.device)
            _assert_edge_index('add_edges', add_edges, num_nodes)
            new_edge_index = torch.cat([base_edge_index, add_edges], dim=1)
        else:
            new_edge_index = base_edge_index.clone()

        if force_undirected:
            new_edge_index = to_undirected(new_edge_index, num_nodes=num_nodes)
        new_edge_index = new_edge_index.to(edge_index.device)
        _assert_edge_index('add_edge_index', new_edge_index, num_nodes)
        return new_edge_index

    for view_id in range(num_views):
        if view_policy == 'drop_only':
            new_edge_index = _drop_view(edge_index)
        elif view_policy == 'drop_add':
            dropped_edge_index = _drop_view(edge_index)
            new_edge_index = _add_view(dropped_edge_index)
        elif view_id % 2 == 0:
            new_edge_index = _drop_view(edge_index)
        else:
            if edge_add == 0.0:
                new_edge_index = _drop_view(edge_index)
            else:
                new_edge_index = _add_view(edge_index)

        assert new_edge_index.device == edge_index.device, 'edge view device mismatch'
        views.append(new_edge_index)

    assert len(views) == num_views + 1, 'unexpected number of perturbed edge views'
    return views


def to_prob(output: torch.Tensor) -> torch.Tensor:
    """
    Convert logits, probabilities, or log_softmax outputs to probabilities.
    """
    _assert_shape('output', output, (None, None))
    _assert_finite('output', output)

    row_sum = output.sum(dim=1)
    is_probability = (output >= -1e-8).all() and torch.allclose(
        row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4
    )

    logsumexp = torch.logsumexp(output, dim=1)
    is_log_prob = torch.allclose(
        logsumexp, torch.zeros_like(logsumexp), atol=1e-4, rtol=1e-4
    )

    if is_probability:
        prob = output
    elif is_log_prob:
        prob = output.exp()
    else:
        prob = F.softmax(output, dim=1)

    prob = prob.clamp_min(0.0)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(EPS)
    _assert_probability('prob', prob, (output.size(0), output.size(1)))
    return prob


class SourceResponseProbe:
    def __init__(self, source_models, device, debug: bool = False):
        assert source_models is not None, 'source_models must not be None'
        self.source_models = list(source_models)
        assert len(self.source_models) > 0, 'source_models must not be empty'
        self.device = torch.device(device)
        self.debug = debug
        self._freeze_sources()

    def _freeze_sources(self):
        for source_model in self.source_models:
            source_model.to(self.device)
            source_model.eval()
            source_model.requires_grad_(False)
            for param in source_model.parameters():
                param.requires_grad_(False)
        self._assert_sources_frozen()

    def _assert_sources_frozen(self):
        for source_id, source_model in enumerate(self.source_models):
            assert not source_model.training, 'source model {} must be in eval mode'.format(source_id)
            for param_id, param in enumerate(source_model.parameters()):
                assert not param.requires_grad, (
                    'source model {} parameter {} must be frozen'.format(source_id, param_id)
                )

    @torch.no_grad()
    def forward_all_views(self, x, edge_views):
        _assert_shape('x', x, (None, None))
        _assert_finite('x', x)
        assert edge_views is not None and len(edge_views) > 0, 'edge_views must not be empty'
        self._freeze_sources()
        self._assert_sources_frozen()

        x_probe = x.to(self.device)
        _assert_shape('x_probe', x_probe, (x.size(0), x.size(1)))
        _assert_finite('x_probe', x_probe)
        _debug_tensor('x_probe', x_probe, self.debug)

        num_nodes = x_probe.size(0)
        num_views = len(edge_views)
        num_sources = len(self.source_models)
        all_source = []
        num_classes = None

        for source_id, source_model in enumerate(self.source_models):
            source_model.eval()
            view_outputs = []
            for view_id, edge_index in enumerate(edge_views):
                edge_index_probe = edge_index.to(self.device)
                _assert_edge_index(
                    'edge_index_probe_{}_{}'.format(source_id, view_id),
                    edge_index_probe,
                    num_nodes,
                )
                _debug_tensor(
                    'edge_index_probe_{}_{}'.format(source_id, view_id),
                    edge_index_probe,
                    self.debug,
                )

                output = source_model(x_probe, edge_index_probe)
                _assert_shape('source_output', output, (num_nodes, None))
                _assert_finite('source_output', output)
                _debug_tensor('source_output', output, self.debug)

                prob = to_prob(output)
                if num_classes is None:
                    num_classes = prob.size(1)
                assert prob.size(1) == num_classes, 'source models disagree on class count'
                _assert_probability('source_prob', prob, (num_nodes, num_classes))
                _debug_tensor('source_prob', prob, self.debug)
                view_outputs.append(prob)

            view_outputs = torch.stack(view_outputs, dim=1).detach()
            _assert_probability(
                'view_outputs',
                view_outputs,
                (num_nodes, num_views, num_classes),
            )
            _debug_tensor('view_outputs', view_outputs, self.debug)
            all_source.append(view_outputs)

        source_probs = torch.stack(all_source, dim=1).detach()
        _assert_probability(
            'source_probs',
            source_probs,
            (num_nodes, num_sources, num_views, num_classes),
        )
        assert not source_probs.requires_grad, 'source_probs must not require gradients'
        self._assert_sources_frozen()
        _debug_tensor('source_probs', source_probs, self.debug)
        return source_probs

    @torch.no_grad()
    def build_features(self, source_probs, edge_index, num_nodes):
        _assert_probability('source_probs', source_probs, (num_nodes, None, None, None))
        _assert_edge_index('edge_index', edge_index, num_nodes)

        num_sources = source_probs.size(1)
        num_views = source_probs.size(2)
        conf = source_probs.max(dim=-1).values
        _assert_shape('conf', conf, (num_nodes, num_sources, num_views))
        _assert_finite('conf', conf)
        _debug_tensor('conf', conf, self.debug)

        mean_conf = conf.mean(dim=-1)
        std_conf = conf.std(dim=-1, unbiased=False)
        preds = source_probs.argmax(dim=-1)
        mode_preds = preds.mode(dim=-1).values
        pred_consistency = (preds == mode_preds.unsqueeze(-1)).float().mean(dim=-1)
        conf_drop = conf[:, :, 0] - mean_conf

        _assert_shape('mean_conf', mean_conf, (num_nodes, num_sources))
        _assert_shape('std_conf', std_conf, (num_nodes, num_sources))
        _assert_shape('preds', preds, (num_nodes, num_sources, num_views))
        _assert_shape('mode_preds', mode_preds, (num_nodes, num_sources))
        _assert_shape('pred_consistency', pred_consistency, (num_nodes, num_sources))
        _assert_shape('conf_drop', conf_drop, (num_nodes, num_sources))
        _assert_finite('mean_conf', mean_conf)
        _assert_finite('std_conf', std_conf)
        _assert_finite('pred_consistency', pred_consistency)
        _assert_finite('conf_drop', conf_drop)
        _debug_tensor('mean_conf', mean_conf, self.debug)
        _debug_tensor('std_conf', std_conf, self.debug)
        _debug_tensor('pred_consistency', pred_consistency, self.debug)
        _debug_tensor('conf_drop', conf_drop, self.debug)

        row, col = edge_index.to(source_probs.device)
        conf0 = conf[:, :, 0]
        neighbor_sum = torch.zeros(
            (num_nodes, num_sources),
            device=source_probs.device,
            dtype=source_probs.dtype,
        )
        degree = torch.zeros(num_nodes, device=source_probs.device, dtype=source_probs.dtype)
        if row.numel() > 0:
            neighbor_sum.index_add_(0, col, conf0[row])
            degree.index_add_(0, col, torch.ones_like(col, dtype=source_probs.dtype))
        _assert_shape('neighbor_sum', neighbor_sum, (num_nodes, num_sources))
        _assert_shape('degree', degree, (num_nodes,))
        _assert_finite('neighbor_sum', neighbor_sum)
        _assert_finite('degree', degree)
        _debug_tensor('neighbor_sum', neighbor_sum, self.debug)
        _debug_tensor('degree', degree, self.debug)

        has_neighbor = degree > 0
        neighbor_mean = torch.zeros_like(neighbor_sum)
        if has_neighbor.any():
            neighbor_mean[has_neighbor] = (
                neighbor_sum[has_neighbor] / degree[has_neighbor].unsqueeze(-1).clamp_min(EPS)
            )
        neighbor_trend = torch.zeros_like(conf0)
        if has_neighbor.any():
            neighbor_trend[has_neighbor] = neighbor_mean[has_neighbor] - conf0[has_neighbor]
        _assert_shape('neighbor_mean', neighbor_mean, (num_nodes, num_sources))
        _assert_shape('neighbor_trend', neighbor_trend, (num_nodes, num_sources))
        _assert_finite('neighbor_mean', neighbor_mean)
        _assert_finite('neighbor_trend', neighbor_trend)
        _debug_tensor('neighbor_mean', neighbor_mean, self.debug)
        _debug_tensor('neighbor_trend', neighbor_trend, self.debug)

        probe_feat = torch.stack(
            [mean_conf, std_conf, pred_consistency, conf_drop, neighbor_trend],
            dim=-1,
        )
        gap_score = (
            (1.0 - mean_conf)
            + std_conf
            + (1.0 - pred_consistency)
            + conf_drop.clamp_min(0.0)
            + neighbor_trend.clamp_min(0.0)
        )
        r_probe = torch.exp(-gap_score)

        _assert_shape('probe_feat', probe_feat, (num_nodes, num_sources, 5))
        _assert_shape('gap_score', gap_score, (num_nodes, num_sources))
        _assert_shape('r_probe', r_probe, (num_nodes, num_sources))
        _assert_finite('probe_feat', probe_feat)
        _assert_finite('gap_score', gap_score)
        _assert_finite('r_probe', r_probe)
        _debug_tensor('probe_feat', probe_feat, self.debug)
        _debug_tensor('gap_score', gap_score, self.debug)
        _debug_tensor('r_probe', r_probe, self.debug)
        return probe_feat, gap_score, r_probe

    @torch.no_grad()
    def __call__(self, x, edge_views, num_nodes):
        source_probs = self.forward_all_views(x, edge_views)
        probe_feat, gap_score, r_probe = self.build_features(
            source_probs,
            edge_views[0].to(source_probs.device),
            num_nodes,
        )
        return source_probs, probe_feat, gap_score, r_probe


@torch.no_grad()
def aggregate_source_response(
    source_probs,
    r_probe,
    temperature: float = 1.0,
    rnode_mode: str = 'max',
):
    """
    Aggregate frozen source responses according to probe reliability.

    Return:
        p_probe: [N, C]
        source_weight: [N, M]
        r_node: [N]
    """
    assert temperature > 0.0, 'temperature must be positive'
    assert rnode_mode in {'max', 'mean', 'topk_mean'}, 'invalid rnode_mode'
    assert source_probs.dim() == 4, 'source_probs must have shape [N, M, S+1, C]'
    num_nodes, num_sources, num_views, num_classes = source_probs.shape
    _assert_probability('source_probs', source_probs, (num_nodes, num_sources, num_views, num_classes))
    _assert_shape('r_probe', r_probe, (num_nodes, num_sources))
    _assert_finite('r_probe', r_probe)
    assert (r_probe >= 0.0).all(), 'r_probe must be non-negative'

    mean_source_probs = source_probs.mean(dim=2)
    _assert_probability('mean_source_probs', mean_source_probs, (num_nodes, num_sources, num_classes))

    source_weight = F.softmax(torch.log(r_probe.clamp_min(EPS)) / temperature, dim=1)
    _assert_probability('source_weight', source_weight, (num_nodes, num_sources))

    p_probe = (source_weight.unsqueeze(-1) * mean_source_probs).sum(dim=1)
    p_probe = p_probe.clamp_min(0.0)
    p_probe = p_probe / p_probe.sum(dim=1, keepdim=True).clamp_min(EPS)
    _assert_probability('p_probe', p_probe, (num_nodes, num_classes))

    if rnode_mode == 'max':
        r_node = r_probe.max(dim=1).values
    elif rnode_mode == 'mean':
        r_node = r_probe.mean(dim=1)
    else:
        topk = min(2, num_sources)
        r_node = torch.topk(r_probe, k=topk, dim=1).values.mean(dim=1)
    _assert_shape('r_node', r_node, (num_nodes,))
    _assert_finite('r_node', r_node)
    assert (r_node >= 0.0).all(), 'r_node must be non-negative'

    return p_probe.detach(), source_weight.detach(), r_node.detach()


__all__ = [
    'build_perturbed_edge_views',
    'to_prob',
    'SourceResponseProbe',
    'aggregate_source_response',
]
