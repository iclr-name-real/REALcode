import torch
import torch.nn.functional as F


EPS = 1e-8


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
        print('[M2] {} shape: {} empty'.format(name, shape))
        return
    stat_tensor = tensor.detach()
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.float()
    print(
        '[M2] {} shape: {} mean={:.6f} min={:.6f} max={:.6f} std={:.6f}'.format(
            name,
            shape,
            stat_tensor.mean().item(),
            stat_tensor.min().item(),
            stat_tensor.max().item(),
            stat_tensor.std(unbiased=False).item(),
        )
    )


def _assert_probability(name, prob, expected_shape):
    _assert_shape(name, prob, expected_shape)
    _assert_finite(name, prob)
    assert (prob >= -1e-6).all(), '{} contains negative probabilities'.format(name)
    row_sum = prob.sum(dim=-1)
    assert torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=1e-4,
        rtol=1e-4,
    ), '{} rows do not sum to 1'.format(name)


def _assert_weight_range(name, weight, expected_shape):
    _assert_shape(name, weight, expected_shape)
    _assert_finite(name, weight)
    assert (weight >= -1e-6).all(), '{} contains values below 0'.format(name)
    assert (weight <= 1.0 + 1e-6).all(), '{} contains values above 1'.format(name)


def _expand_node_weight(name, weight, num_nodes, num_sources):
    if weight.dim() == 1:
        _assert_weight_range(name, weight, (num_nodes,))
        return weight.unsqueeze(1)
    if weight.dim() == 2:
        if weight.size(1) == 1:
            _assert_weight_range(name, weight, (num_nodes, 1))
            return weight
        _assert_weight_range(name, weight, (num_nodes, num_sources))
        return weight
    raise AssertionError('{} must have shape [N], [N, 1], or [N, M]'.format(name))


@torch.no_grad()
def robust_normalize(x, low=0.05, high=0.95, eps=1e-8):
    x = x.detach().float()
    _assert_finite('robust_normalize_input', x)
    assert x.numel() > 0, 'robust_normalize input must be non-empty'
    lo = torch.quantile(x.reshape(-1), low)
    hi = torch.quantile(x.reshape(-1), high)
    x_norm = (x - lo) / (hi - lo + eps)
    x_norm = x_norm.clamp(0.0, 1.0)
    _assert_weight_range('robust_normalize_output', x_norm, tuple(x.shape))
    return x_norm.detach()


@torch.no_grad()
def rank_normalize(x):
    x = x.detach().float()
    _assert_finite('rank_normalize_input', x)
    assert x.numel() > 0, 'rank_normalize input must be non-empty'
    flat = x.reshape(-1)
    if flat.numel() == 1:
        x_norm = torch.zeros_like(flat)
    else:
        order = torch.argsort(flat, stable=True)
        ranks = torch.empty_like(order, dtype=x.dtype)
        ranks[order] = torch.arange(flat.numel(), device=x.device, dtype=x.dtype)
        x_norm = ranks / float(flat.numel() - 1)
    x_norm = x_norm.reshape_as(x).clamp(0.0, 1.0)
    _assert_weight_range('rank_normalize_output', x_norm, tuple(x.shape))
    return x_norm.detach()


@torch.no_grad()
def knn_pseudo_label(feat_output, mem_fea, mem_cls, K, debug=False):
    """
    Equivalent to GraphATA memory-bank nearest-neighbor pseudo-label generation.

    Return:
        pseudo_y: [N]
        pseudo_dist: [N, C]
    """
    _assert_shape('feat_output', feat_output, (None, None))
    _assert_shape('mem_fea', mem_fea, (None, feat_output.size(1)))
    _assert_shape('mem_cls', mem_cls, (mem_fea.size(0), None))
    _assert_finite('feat_output', feat_output)
    _assert_finite('mem_fea', mem_fea)
    _assert_finite('mem_cls', mem_cls)
    assert feat_output.device == mem_fea.device, 'feat_output and mem_fea must be on same device'
    assert feat_output.device == mem_cls.device, 'feat_output and mem_cls must be on same device'
    assert K > 0, 'K must be positive'
    assert K + 1 <= mem_fea.size(0), 'K + 1 must not exceed memory bank size'

    num_nodes = feat_output.size(0)
    num_classes = mem_cls.size(1)

    feat_norm = F.normalize(feat_output, dim=1)
    mem_fea_norm = F.normalize(mem_fea, dim=1)
    distance = feat_norm @ mem_fea_norm.T
    _assert_shape('distance', distance, (num_nodes, mem_fea.size(0)))
    _assert_finite('distance', distance)
    _debug_tensor('distance', distance, debug)

    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K + 1)
    idx_near = idx_near[:, 1:]
    _assert_shape('idx_near', idx_near, (num_nodes, K))
    assert idx_near.dtype == torch.long, 'idx_near must have dtype torch.long'
    _debug_tensor('idx_near', idx_near, debug)

    pseudo_dist = torch.mean(mem_cls[idx_near], dim=1)
    pseudo_dist = pseudo_dist / (pseudo_dist.sum(dim=1, keepdim=True) + EPS)
    pseudo_y = pseudo_dist.argmax(dim=1)
    _assert_probability('pseudo_dist', pseudo_dist, (num_nodes, num_classes))
    _assert_shape('pseudo_y', pseudo_y, (num_nodes,))
    _debug_tensor('pseudo_dist', pseudo_dist, debug)
    _debug_tensor('pseudo_y', pseudo_y, debug)
    return pseudo_y.detach(), pseudo_dist.detach()


@torch.no_grad()
def compute_wrong_score(probs, pseudo_y, pseudo_dist=None, mode='confidence'):
    probs = probs.detach().float().clamp(EPS, 1.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
    _assert_probability('wrong_score_probs', probs, (probs.size(0), None))
    _assert_shape('wrong_score_pseudo_y', pseudo_y, (probs.size(0),))
    assert pseudo_y.dtype == torch.long, 'pseudo_y must have dtype torch.long'

    if mode == 'argmax':
        wrong_score = (probs.argmax(dim=1) != pseudo_y).float()
    elif mode == 'confidence':
        py = probs.gather(1, pseudo_y.view(-1, 1)).squeeze(1)
        wrong_score = 1.0 - py
    elif mode == 'soft':
        assert pseudo_dist is not None, 'pseudo_dist is required for soft wrong signal'
        pseudo_dist = pseudo_dist.detach().float().clamp(EPS, 1.0)
        pseudo_dist = pseudo_dist / pseudo_dist.sum(dim=1, keepdim=True).clamp_min(EPS)
        _assert_probability('wrong_score_pseudo_dist', pseudo_dist, probs.shape)
        wrong_score = 1.0 - (probs * pseudo_dist).sum(dim=1)
    elif mode == 'kl':
        assert pseudo_dist is not None, 'pseudo_dist is required for kl wrong signal'
        pseudo_dist = pseudo_dist.detach().float().clamp(EPS, 1.0)
        pseudo_dist = pseudo_dist / pseudo_dist.sum(dim=1, keepdim=True).clamp_min(EPS)
        _assert_probability('wrong_score_pseudo_dist', pseudo_dist, probs.shape)
        wrong_score = (pseudo_dist * (pseudo_dist.log() - probs.log())).sum(dim=1)
    else:
        raise ValueError('Unknown wrong signal mode: {}'.format(mode))

    wrong_score = wrong_score.float().clamp(0.0, 1.0)
    _assert_weight_range('wrong_score', wrong_score, (probs.size(0),))
    return wrong_score.detach()


class WrongEventTracker:
    def __init__(
        self,
        num_nodes,
        num_sources,
        device,
        debug=False,
        wrong_signal='confidence',
        wrong_momentum=0.9,
        expert_wrong_signal='argmax',
        expert_wrong_momentum=0.9,
    ):
        assert num_nodes > 0, 'num_nodes must be positive'
        assert num_sources > 0, 'num_sources must be positive'
        assert wrong_signal in ('argmax', 'confidence', 'soft', 'kl'), 'invalid wrong_signal'
        assert 0.0 <= wrong_momentum < 1.0, 'wrong_momentum must be in [0, 1)'
        assert expert_wrong_signal in ('argmax', 'confidence'), 'invalid expert_wrong_signal'
        assert 0.0 <= expert_wrong_momentum < 1.0, 'expert_wrong_momentum must be in [0, 1)'
        self.num_nodes = int(num_nodes)
        self.num_sources = int(num_sources)
        self.device = torch.device(device)
        self.debug = debug
        self.wrong_signal = wrong_signal
        self.wrong_momentum = float(wrong_momentum)
        self.expert_wrong_signal = expert_wrong_signal
        self.expert_wrong_momentum = float(expert_wrong_momentum)
        self.node_wrong = torch.zeros(self.num_nodes, device=self.device)
        self.expert_wrong = torch.zeros(self.num_nodes, self.num_sources, device=self.device)
        self.count = 0
        self.expert_count = 0
        _assert_shape('node_wrong', self.node_wrong, (self.num_nodes,))
        _assert_shape('expert_wrong', self.expert_wrong, (self.num_nodes, self.num_sources))
        _debug_tensor('node_wrong', self.node_wrong, self.debug)
        _debug_tensor('expert_wrong', self.expert_wrong, self.debug)

    @torch.no_grad()
    def update_node(self, probs, pseudo_y, pseudo_dist=None):
        probs = probs.detach().to(self.device)
        pseudo_y = pseudo_y.detach().to(self.device)
        if pseudo_dist is not None:
            pseudo_dist = pseudo_dist.detach().to(self.device)
        _assert_probability('probs', probs, (self.num_nodes, None))
        _assert_shape('pseudo_y', pseudo_y, (self.num_nodes,))
        assert pseudo_y.dtype == torch.long, 'pseudo_y must have dtype torch.long'

        wrong_score = compute_wrong_score(
            probs,
            pseudo_y,
            pseudo_dist=pseudo_dist,
            mode=self.wrong_signal,
        )
        _assert_shape('node_wrong_score', wrong_score, (self.num_nodes,))
        _assert_finite('node_wrong_score', wrong_score)
        if self.count == 0:
            self.node_wrong = wrong_score.clone()
        else:
            self.node_wrong = (
                self.wrong_momentum * self.node_wrong
                + (1.0 - self.wrong_momentum) * wrong_score
            )
        self.node_wrong = self.node_wrong.clamp(0.0, 1.0)
        self.count += 1
        _debug_tensor('node_wrong_score', wrong_score, self.debug)
        _debug_tensor('node_wrong_ema', self.node_wrong, self.debug)

    @torch.no_grad()
    def update_expert(self, source_probs, pseudo_y):
        source_probs = source_probs.detach().to(self.device)
        pseudo_y = pseudo_y.detach().to(self.device)
        _assert_probability('source_probs', source_probs, (self.num_nodes, self.num_sources, None, None))
        _assert_shape('pseudo_y', pseudo_y, (self.num_nodes,))
        assert pseudo_y.dtype == torch.long, 'pseudo_y must have dtype torch.long'

        source_mean = source_probs.mean(dim=2)
        _assert_probability('source_mean', source_mean, (self.num_nodes, self.num_sources, source_probs.size(-1)))
        if self.expert_wrong_signal == 'argmax':
            expert_pred = source_mean.argmax(dim=-1)
            wrong_score = (expert_pred != pseudo_y.unsqueeze(1)).float()
        else:
            pseudo_index = pseudo_y.view(self.num_nodes, 1, 1).expand(-1, self.num_sources, 1)
            pseudo_prob = source_mean.gather(dim=2, index=pseudo_index).squeeze(-1)
            wrong_score = 1.0 - pseudo_prob
        wrong_score = wrong_score.float().clamp(0.0, 1.0)
        _assert_weight_range('expert_wrong_score', wrong_score, (self.num_nodes, self.num_sources))
        if self.expert_count == 0:
            self.expert_wrong = wrong_score.clone()
        else:
            self.expert_wrong = (
                self.expert_wrong_momentum * self.expert_wrong
                + (1.0 - self.expert_wrong_momentum) * wrong_score
            )
        self.expert_wrong = self.expert_wrong.clamp(0.0, 1.0)
        self.expert_count += 1
        _debug_tensor('expert_wrong_score', wrong_score, self.debug)
        _debug_tensor('expert_wrong_ema', self.expert_wrong, self.debug)

    @torch.no_grad()
    def get_node_wrong_rate(self):
        node_wrong_rate = self.node_wrong.clamp(0.0, 1.0)
        _assert_weight_range('node_wrong_rate', node_wrong_rate, (self.num_nodes,))
        _debug_tensor('node_wrong_rate', node_wrong_rate, self.debug)
        return node_wrong_rate.detach()

    @torch.no_grad()
    def get_expert_wrong_rate(self):
        expert_wrong_rate = self.expert_wrong.clamp(0.0, 1.0)
        _assert_weight_range('expert_wrong_rate', expert_wrong_rate, (self.num_nodes, self.num_sources))
        _debug_tensor('expert_wrong_rate', expert_wrong_rate, self.debug)
        return expert_wrong_rate.detach()


@torch.no_grad()
def fallback_wrong_event_weights(wrong_rate, debug=False, wrong_norm='none'):
    """
    Build fallback clean/noisy/difficult weights from wrong-event rate.

    wrong_rate can be [N] or [N, M]. Returned tensors keep the same shape.
    """
    assert wrong_rate.dim() in (1, 2), 'wrong_rate must have shape [N] or [N, M]'
    assert wrong_norm in {'none', 'robust', 'rank'}, 'invalid wrong_norm'
    wrong_rate = wrong_rate.detach().float()
    _assert_finite('wrong_rate', wrong_rate)
    wrong_rate = wrong_rate.clamp(0.0, 1.0)

    if wrong_norm == 'none':
        wrong_rate_for_weight = wrong_rate
    elif wrong_norm == 'robust':
        wrong_rate_for_weight = robust_normalize(wrong_rate)
    else:
        wrong_rate_for_weight = rank_normalize(wrong_rate)
    _assert_weight_range('wrong_rate_for_weight', wrong_rate_for_weight, tuple(wrong_rate.shape))

    q_clean = (1.0 - wrong_rate_for_weight).clamp(0.0, 1.0)
    q_noise = wrong_rate_for_weight.clamp(0.0, 1.0)
    q_diff = (4.0 * wrong_rate_for_weight * (1.0 - wrong_rate_for_weight)).clamp(0.0, 1.0)

    expected_shape = tuple(wrong_rate.shape)
    _assert_weight_range('q_clean', q_clean, expected_shape)
    _assert_weight_range('q_noise', q_noise, expected_shape)
    _assert_weight_range('q_diff', q_diff, expected_shape)
    _debug_tensor('wrong_rate', wrong_rate, debug)
    _debug_tensor('wrong_rate_for_weight', wrong_rate_for_weight, debug)
    _debug_tensor('q_clean', q_clean, debug)
    _debug_tensor('q_noise', q_noise, debug)
    _debug_tensor('q_diff', q_diff, debug)
    return q_clean.detach(), q_noise.detach(), q_diff.detach()


@torch.no_grad()
def build_dynamic_weights(wrong_rate, force_clean=False, debug=False, wrong_norm='none'):
    """
    Build node-level dynamic weights for PHTA.

    If force_clean is True, returns all-clean weights for warmup:
        q_clean = 1, q_noise = 0, q_diff = 0.
    Otherwise, uses fallback_wrong_event_weights.
    """
    assert wrong_norm in {'none', 'robust', 'rank'}, 'invalid wrong_norm'
    assert wrong_rate.dim() == 1, 'wrong_rate must have shape [N] for node-level weights'
    wrong_rate = wrong_rate.detach().float()
    _assert_finite('wrong_rate', wrong_rate)

    if force_clean:
        q_clean = torch.ones_like(wrong_rate)
        q_noise = torch.zeros_like(wrong_rate)
        q_diff = torch.zeros_like(wrong_rate)
        _assert_weight_range('q_clean', q_clean, (wrong_rate.size(0),))
        _assert_weight_range('q_noise', q_noise, (wrong_rate.size(0),))
        _assert_weight_range('q_diff', q_diff, (wrong_rate.size(0),))
        _debug_tensor('q_clean', q_clean, debug)
        _debug_tensor('q_noise', q_noise, debug)
        _debug_tensor('q_diff', q_diff, debug)
        return q_clean.detach(), q_noise.detach(), q_diff.detach()

    return fallback_wrong_event_weights(wrong_rate, debug=debug, wrong_norm=wrong_norm)


def _weighted_beta_moments(x, weight, eps):
    weight_sum = weight.sum().clamp_min(eps)
    mean = (weight * x).sum() / weight_sum
    var = (weight * (x - mean) ** 2).sum() / weight_sum
    mean = mean.clamp(eps, 1.0 - eps)
    max_var = (mean * (1.0 - mean)).clamp_min(eps)
    assert torch.isfinite(var), 'beta variance is not finite'
    assert var > eps, 'beta variance is too small'
    assert var < max_var, 'beta variance is invalid for beta distribution'
    scale = max_var / var - 1.0
    alpha = mean * scale
    beta = (1.0 - mean) * scale
    assert torch.isfinite(alpha) and torch.isfinite(beta), 'beta parameters are not finite'
    assert alpha > eps and beta > eps, 'beta parameters must be positive'
    return alpha, beta


def _beta_pdf(x, alpha, beta, eps):
    x = x.clamp(eps, 1.0 - eps)
    log_pdf = (
        (alpha - 1.0) * torch.log(x)
        + (beta - 1.0) * torch.log1p(-x)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )
    pdf = torch.exp(log_pdf).clamp_min(0.0)
    assert torch.isfinite(pdf).all(), 'beta pdf contains NaN or Inf'
    return pdf


def _beta_cdf(x, alpha, beta, eps):
    try:
        from scipy.special import betainc
    except Exception as exc:
        raise RuntimeError('scipy.special.betainc is unavailable: {}'.format(exc))

    x_np = x.detach().clamp(eps, 1.0 - eps).cpu().numpy()
    cdf_np = betainc(float(alpha.detach().cpu()), float(beta.detach().cpu()), x_np)
    cdf = torch.as_tensor(cdf_np, device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
    assert torch.isfinite(cdf).all(), 'beta cdf contains NaN or Inf'
    return cdf


def _fit_two_component_beta_mixture(x, max_iter, eps):
    assert x.dim() == 1, 'x must have shape [Nc]'
    assert x.numel() > 0, 'x must not be empty'
    x = x.detach().float().clamp(eps, 1.0 - eps)
    _assert_finite('bmm_x', x)

    median = x.median()
    resp_clean = (x <= median).float()
    resp_noise = 1.0 - resp_clean
    if resp_clean.sum() < 2 or resp_noise.sum() < 2:
        resp_clean = (1.0 - x).clamp(eps, 1.0)
        resp_noise = x.clamp(eps, 1.0)
    resp_sum = (resp_clean + resp_noise).clamp_min(eps)
    resp_clean = resp_clean / resp_sum
    resp_noise = resp_noise / resp_sum

    converged = False
    params = None

    for _ in range(max_iter):
        alpha_clean, beta_clean = _weighted_beta_moments(x, resp_clean, eps)
        alpha_noise, beta_noise = _weighted_beta_moments(x, resp_noise, eps)
        pi_clean = resp_clean.mean().clamp(eps, 1.0 - eps)
        pi_noise = (1.0 - pi_clean).clamp(eps, 1.0 - eps)

        pdf_clean = _beta_pdf(x, alpha_clean, beta_clean, eps)
        pdf_noise = _beta_pdf(x, alpha_noise, beta_noise, eps)
        mix_clean = pi_clean * pdf_clean
        mix_noise = pi_noise * pdf_noise
        denom = mix_clean + mix_noise
        assert torch.isfinite(denom).all(), 'bmm mixture density contains NaN or Inf'
        assert (denom > eps).any(), 'bmm mixture density is all zero'
        denom = denom.clamp_min(eps)

        resp_clean_new = mix_clean / denom
        resp_noise_new = mix_noise / denom
        _assert_finite('resp_clean_new', resp_clean_new)
        _assert_finite('resp_noise_new', resp_noise_new)

        resp_delta = torch.max(torch.abs(resp_clean_new - resp_clean))
        assert torch.isfinite(resp_delta), 'bmm responsibility delta is NaN or Inf'
        resp_clean = resp_clean_new
        resp_noise = resp_noise_new
        params = (alpha_clean, beta_clean, alpha_noise, beta_noise, pi_clean, pi_noise)
        if resp_delta < eps:
            converged = True
            break

    assert converged, 'bmm did not converge'

    alpha_clean, beta_clean, alpha_noise, beta_noise, pi_clean, pi_noise = params
    mean_clean = alpha_clean / (alpha_clean + beta_clean)
    mean_noise = alpha_noise / (alpha_noise + beta_noise)
    if mean_clean > mean_noise:
        alpha_clean, alpha_noise = alpha_noise, alpha_clean
        beta_clean, beta_noise = beta_noise, beta_clean
        pi_clean, pi_noise = pi_noise, pi_clean
        resp_clean, resp_noise = resp_noise, resp_clean
        mean_clean, mean_noise = mean_noise, mean_clean
    assert mean_clean <= mean_noise, 'clean beta component must have smaller mean'

    lambda_clean = _beta_cdf(x, alpha_clean, beta_clean, eps)
    lambda_noise = _beta_cdf(x, alpha_noise, beta_noise, eps)
    q_clean = resp_clean.clamp(0.0, 1.0)
    q_noise = resp_noise.clamp(0.0, 1.0)
    q_diff = (q_clean * lambda_clean + q_noise * (1.0 - lambda_noise)).clamp(0.0, 1.0)

    _assert_weight_range('bmm_q_clean', q_clean, (x.numel(),))
    _assert_weight_range('bmm_q_noise', q_noise, (x.numel(),))
    _assert_weight_range('bmm_q_diff', q_diff, (x.numel(),))
    return q_clean, q_noise, q_diff


@torch.no_grad()
def classwise_bmm_wrong_event_weights(
    wrong_rate,
    pseudo_y,
    num_classes,
    min_class_count=20,
    max_iter=10,
    eps=1e-4,
    debug=False,
):
    """
    Build clean/noisy/difficult weights with class-wise two-component BMM.

    Any class-level BMM failure falls back to fallback_wrong_event_weights for that class.
    """
    _assert_weight_range('wrong_rate', wrong_rate, (None,))
    _assert_shape('pseudo_y', pseudo_y, (wrong_rate.size(0),))
    assert pseudo_y.dtype == torch.long, 'pseudo_y must have dtype torch.long'
    assert num_classes > 0, 'num_classes must be positive'
    assert min_class_count > 0, 'min_class_count must be positive'
    assert max_iter > 0, 'max_iter must be positive'
    assert eps > 0.0, 'eps must be positive'
    assert (pseudo_y >= 0).all() and (pseudo_y < num_classes).all(), (
        'pseudo_y must be in [0, num_classes)'
    )

    wrong_rate = wrong_rate.detach().float().clamp(0.0, 1.0)
    pseudo_y = pseudo_y.detach()
    num_nodes = wrong_rate.size(0)
    q_clean = torch.empty_like(wrong_rate)
    q_noise = torch.empty_like(wrong_rate)
    q_diff = torch.empty_like(wrong_rate)
    success_classes = []
    fallback_classes = []
    fallback_reasons = {}

    for class_id in range(num_classes):
        class_mask = pseudo_y == class_id
        class_count = int(class_mask.sum().item())
        if class_count == 0:
            continue

        class_wrong = wrong_rate[class_mask]
        if class_count < min_class_count:
            class_q_clean, class_q_noise, class_q_diff = fallback_wrong_event_weights(class_wrong)
            fallback_classes.append(class_id)
            fallback_reasons[class_id] = 'class_count {} < min_class_count {}'.format(
                class_count, min_class_count
            )
        else:
            try:
                class_q_clean, class_q_noise, class_q_diff = _fit_two_component_beta_mixture(
                    class_wrong,
                    max_iter=max_iter,
                    eps=eps,
                )
                success_classes.append(class_id)
            except Exception as exc:
                class_q_clean, class_q_noise, class_q_diff = fallback_wrong_event_weights(class_wrong)
                fallback_classes.append(class_id)
                fallback_reasons[class_id] = str(exc)

        q_clean[class_mask] = class_q_clean.to(device=wrong_rate.device, dtype=wrong_rate.dtype)
        q_noise[class_mask] = class_q_noise.to(device=wrong_rate.device, dtype=wrong_rate.dtype)
        q_diff[class_mask] = class_q_diff.to(device=wrong_rate.device, dtype=wrong_rate.dtype)

    _assert_weight_range('q_clean_bmm', q_clean, (num_nodes,))
    _assert_weight_range('q_noise_bmm', q_noise, (num_nodes,))
    _assert_weight_range('q_diff_bmm', q_diff, (num_nodes,))
    _debug_tensor('q_clean_bmm', q_clean, debug)
    _debug_tensor('q_noise_bmm', q_noise, debug)
    _debug_tensor('q_diff_bmm', q_diff, debug)
    info = {
        'success_classes': success_classes,
        'fallback_classes': fallback_classes,
        'fallback_reasons': fallback_reasons,
    }
    return q_clean.detach(), q_noise.detach(), q_diff.detach(), info


@torch.no_grad()
def build_reliability_difficulty(
    r_probe,
    gap_score,
    q_clean,
    q_diff,
    expert_wrong_rate=None,
    reliability_mode='legacy',
    gap_floor_mode='none',
    gap_floor_mix=0.4,
    source_prune_mode='none',
    source_prune_floor=0.0,
    debug=False,
):
    """
    Fuse SPRE reliability and PHTA wrong-event weights.

    Return:
        reliability: [N, M]
        difficulty: [N, M]
    """
    _assert_shape('r_probe', r_probe, (None, None))
    num_nodes, num_sources = r_probe.shape
    _assert_shape('gap_score', gap_score, (num_nodes, num_sources))
    _assert_finite('r_probe', r_probe)
    _assert_finite('gap_score', gap_score)
    assert (r_probe >= -1e-6).all(), 'r_probe must be non-negative'
    assert (r_probe <= 1.0 + 1e-6).all(), 'r_probe must not exceed 1'
    assert (gap_score >= -1e-6).all(), 'gap_score must be non-negative'
    assert gap_floor_mode in ('none', 'max', 'mix'), 'invalid gap_floor_mode'
    assert reliability_mode in (
        'legacy', 'probe', 'probe_expert', 'pwed_rescue'
    ), 'invalid reliability_mode'
    assert 0.0 <= gap_floor_mix <= 1.0, 'gap_floor_mix must be in [0, 1]'
    assert source_prune_mode in ('none', 'global_top1', 'local_top1'), 'invalid source_prune_mode'
    assert 0.0 <= source_prune_floor <= 1.0, 'source_prune_floor must be in [0, 1]'

    q_clean_expand = _expand_node_weight('q_clean', q_clean.detach().float(), num_nodes, num_sources)
    q_diff_expand = _expand_node_weight('q_diff', q_diff.detach().float(), num_nodes, num_sources)
    if q_diff_expand.size(1) == 1:
        q_diff_for_difficulty = q_diff_expand.expand(num_nodes, num_sources)
    else:
        q_diff_for_difficulty = q_diff_expand

    if expert_wrong_rate is None:
        expert_clean = torch.ones_like(r_probe)
    else:
        expert_wrong_rate = expert_wrong_rate.detach().float()
        _assert_weight_range('expert_wrong_rate', expert_wrong_rate, (num_nodes, num_sources))
        expert_clean = (1.0 - expert_wrong_rate).clamp(0.0, 1.0)
    _assert_weight_range('expert_clean', expert_clean, (num_nodes, num_sources))

    if reliability_mode == 'legacy':
        reliability = q_clean_expand * r_probe * expert_clean
    elif reliability_mode == 'probe':
        
        
        
        reliability = r_probe.clone()
    elif reliability_mode == 'probe_expert':
        reliability = r_probe * expert_clean
    else:
        
        
        
        pseudo_uncertainty = torch.maximum(
            (1.0 - q_clean_expand).clamp(0.0, 1.0),
            q_diff_for_difficulty.clamp(0.0, 1.0),
        )
        expert_trust = expert_clean + (1.0 - expert_clean) * pseudo_uncertainty
        _assert_weight_range('pseudo_uncertainty', pseudo_uncertainty, (num_nodes, num_sources))
        _assert_weight_range('expert_trust', expert_trust, (num_nodes, num_sources))
        reliability = r_probe * expert_trust
        _debug_tensor('pseudo_uncertainty', pseudo_uncertainty, debug)
        _debug_tensor('expert_trust', expert_trust, debug)
    reliability = reliability.clamp(0.0, 1.0)
    if source_prune_mode != 'none':
        if source_prune_mode == 'global_top1':
            source_quality = reliability.mean(dim=0)
            _assert_weight_range('source_quality', source_quality, (num_sources,))
            top_source = int(torch.argmax(source_quality).item())
            source_mask = torch.zeros_like(source_quality)
            source_mask[top_source] = 1.0
            source_mask = source_mask.unsqueeze(0).expand_as(reliability)
        else:
            top_source = torch.argmax(reliability, dim=1)
            source_mask = torch.zeros_like(reliability)
            source_mask.scatter_(1, top_source.view(-1, 1), 1.0)
        prune_multiplier = source_prune_floor + (1.0 - source_prune_floor) * source_mask
        _assert_weight_range('source_prune_multiplier', prune_multiplier, (num_nodes, num_sources))
        reliability = (reliability * prune_multiplier).clamp(0.0, 1.0)
        _debug_tensor('source_prune_multiplier', prune_multiplier, debug)
    if gap_floor_mode == 'none':
        q_diff_final = q_diff_expand
        gap_factor = gap_score
    else:
        gap_node = robust_normalize(gap_score.mean(dim=1))
        gap_pair = robust_normalize(gap_score.reshape(-1)).reshape_as(gap_score)
        gap_node_expand = gap_node.unsqueeze(1).expand_as(q_diff_for_difficulty)
        if gap_floor_mode == 'max':
            q_diff_final = torch.maximum(q_diff_for_difficulty, gap_node_expand)
        else:
            q_diff_final = (
                (1.0 - gap_floor_mix) * q_diff_for_difficulty
                + gap_floor_mix * gap_node_expand
            )
        q_diff_final = q_diff_final.clamp(0.0, 1.0)
        gap_factor = gap_pair
        _assert_weight_range('gap_node', gap_node, (num_nodes,))
        _assert_weight_range('gap_pair', gap_pair, (num_nodes, num_sources))
        _assert_weight_range('q_diff_final', q_diff_final, (num_nodes, num_sources))
        _debug_tensor('gap_node', gap_node, debug)
        _debug_tensor('gap_pair', gap_pair, debug)
        _debug_tensor('q_diff_final', q_diff_final, debug)
    difficulty = q_diff_final * (1.0 + gap_factor)

    _assert_shape('reliability', reliability, (num_nodes, num_sources))
    _assert_shape('difficulty', difficulty, (num_nodes, num_sources))
    _assert_finite('reliability', reliability)
    _assert_finite('difficulty', difficulty)
    _assert_weight_range('reliability', reliability, (num_nodes, num_sources))
    assert (difficulty >= -1e-6).all(), 'difficulty must be non-negative'
    _debug_tensor('r_probe', r_probe, debug)
    _debug_tensor('gap_score', gap_score, debug)
    _debug_tensor('expert_clean', expert_clean, debug)
    if debug:
        print('[M2] reliability_mode={}'.format(reliability_mode))
    _debug_tensor('reliability', reliability, debug)
    _debug_tensor('difficulty', difficulty, debug)
    return reliability.detach(), difficulty.detach()


__all__ = [
    'knn_pseudo_label',
    'WrongEventTracker',
    'fallback_wrong_event_weights',
    'robust_normalize',
    'rank_normalize',
    'build_dynamic_weights',
    'classwise_bmm_wrong_event_weights',
    'build_reliability_difficulty',
]
