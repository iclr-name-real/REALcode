import torch
import torch.nn as nn
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


def _assert_probability(name, prob, expected_shape, atol=1e-4):
    _assert_shape(name, prob, expected_shape)
    _assert_finite(name, prob)
    assert (prob >= -1e-6).all(), '{} contains negative probabilities'.format(name)
    row_sum = prob.sum(dim=-1)
    assert torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=atol,
        rtol=atol,
    ), '{} rows do not sum to 1'.format(name)


def _assert_unit_interval(name, tensor, expected_shape):
    _assert_shape(name, tensor, expected_shape)
    _assert_finite(name, tensor)
    assert (tensor >= -1e-6).all(), '{} contains values below 0'.format(name)
    assert (tensor <= 1.0 + 1e-6).all(), '{} contains values above 1'.format(name)


def _debug_tensor(name, tensor, debug, prefix='[M3]'):
    if not debug:
        return
    stat_tensor = tensor.detach()
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.float()
    print(
        '{} {} shape: {} mean={:.6f} min={:.6f} max={:.6f}'.format(
            prefix,
            name,
            list(tensor.shape),
            stat_tensor.mean().item(),
            stat_tensor.min().item(),
            stat_tensor.max().item(),
        )
    )


def _normalize_probability(prob):
    _assert_finite('prob_before_normalize', prob)
    prob = prob.clamp_min(0.0)
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(EPS)
    _assert_finite('prob_after_normalize', prob)
    return prob


def _temperature_calibrate_probability(prob, temperature, name):
    assert temperature > 0.0, '{} temperature must be positive'.format(name)
    _assert_shape(name, prob, tuple(None for _ in range(prob.dim())))
    _assert_finite(name, prob)
    calibrated = F.softmax(torch.log(prob.clamp_min(EPS)) / float(temperature), dim=-1)
    calibrated = _normalize_probability(calibrated)
    _assert_probability(
        '{}_calibrated'.format(name),
        calibrated,
        tuple(None for _ in range(calibrated.dim())),
    )
    return calibrated


def _entropy_regularization(prob):
    _assert_probability('entropy_prob', prob, (None, None))
    entropy_loss = torch.mean(-torch.sum(prob * torch.log(prob.clamp_min(EPS)), dim=1))
    mean_prob = prob.mean(dim=0)
    div_loss = torch.sum(mean_prob * torch.log(mean_prob.clamp_min(EPS)))
    loss_reg = entropy_loss + div_loss
    _assert_finite('loss_reg', loss_reg)
    return loss_reg


class PriorMoEGate(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        ata_floor: float = 0.10,
        score_mode: str = 'legacy',
        source_sharpness: float = 1.0,
        difficulty_weight: float = 1.0,
        confidence_weight: float = 1.0,
        use_sparsemax: bool = True,
        debug: bool = False,
    ):
        super(PriorMoEGate, self).__init__()
        assert beta >= 0.0, 'beta must be non-negative'
        assert 0.0 <= ata_floor <= 1.0, 'ata_floor must be in [0, 1]'
        assert source_sharpness > 0.0, 'source_sharpness must be positive'
        assert difficulty_weight >= 0.0, 'difficulty_weight must be non-negative'
        assert confidence_weight >= 0.0, 'confidence_weight must be non-negative'
        assert score_mode in (
            'legacy',
            'difficulty_to_source',
            'source_trust',
            'idea_entropy',
            'idea_joint',
            'idea_calibrated',
            'pwed_balanced',
        ), 'invalid score_mode'
        self.beta = float(beta)
        self.ata_floor = float(ata_floor)
        self.score_mode = score_mode
        self.source_sharpness = float(source_sharpness)
        self.difficulty_weight = float(difficulty_weight)
        self.confidence_weight = float(confidence_weight)
        self.debug = debug
        self.use_sparsemax = bool(use_sparsemax)
        self.sparsemax = None
        if self.use_sparsemax:
            try:
                from layer import Sparsemax
                self.sparsemax = Sparsemax(dim=1)
            except Exception:
                self.sparsemax = None
                self.use_sparsemax = False

    def _normalize_scores(self, scores):
        _assert_shape('moe_scores', scores, (None, None))
        _assert_finite('moe_scores', scores)
        if self.use_sparsemax and self.sparsemax is not None:
            gate = self.sparsemax(scores)
            gate = _normalize_probability(gate)
        else:
            gate = F.softmax(scores, dim=1)
        _assert_probability('gate_before_floor', gate, (scores.size(0), scores.size(1)))
        return gate

    def _build_scores(
        self,
        reliability,
        difficulty,
        source_entropy=None,
        p_ata=None,
        source_mean_probs=None,
    ):
        max_reliability = reliability.max(dim=1, keepdim=True).values
        mean_difficulty = difficulty.mean(dim=1, keepdim=True)

        if self.score_mode == 'legacy':
            source_scores = torch.log(reliability.clamp_min(EPS)) - self.beta * difficulty
            ata_scores = (
                torch.log((1.0 - max_reliability).clamp_min(EPS))
                + self.beta * mean_difficulty
            )
        elif self.score_mode == 'difficulty_to_source':
            source_scores = torch.log(reliability.clamp_min(EPS)) + self.beta * difficulty
            ata_scores = (
                torch.log((1.0 - max_reliability).clamp_min(EPS))
                - self.beta * mean_difficulty
            )
        elif self.score_mode == 'source_trust':
            difficulty_unit = mean_difficulty.clamp(0.0, 1.0)
            source_strength = reliability * (1.0 + self.beta * difficulty.clamp(0.0, 1.0))
            ata_strength = (1.0 - difficulty_unit) * (1.0 - max_reliability)
            source_scores = torch.log(source_strength.clamp_min(EPS))
            ata_scores = torch.log(ata_strength.clamp_min(EPS))
        elif self.score_mode == 'idea_entropy':
            _assert_shape('source_entropy', source_entropy, reliability.shape)
            _assert_finite('source_entropy', source_entropy)
            assert (source_entropy >= -1e-6).all(), 'source_entropy must be non-negative'
            source_entropy = source_entropy.clamp_min(0.0)
            mean_source_entropy = source_entropy.mean(dim=1, keepdim=True)
            source_scores = torch.log(reliability.clamp_min(EPS)) - self.beta * source_entropy
            ata_scores = (
                torch.log((1.0 - max_reliability).clamp_min(EPS))
                + self.beta * mean_source_entropy
            )
        elif self.score_mode in ('idea_joint', 'idea_calibrated'):
            _assert_shape('source_entropy', source_entropy, reliability.shape)
            _assert_finite('source_entropy', source_entropy)
            assert (source_entropy >= -1e-6).all(), 'source_entropy must be non-negative'
            source_entropy = source_entropy.clamp_min(0.0)
            mean_source_entropy = source_entropy.mean(dim=1, keepdim=True)
            difficulty_unit = difficulty / (1.0 + difficulty)
            mean_difficulty_unit = difficulty_unit.mean(dim=1, keepdim=True)
            source_scores = (
                torch.log(reliability.clamp_min(EPS))
                - self.beta * source_entropy
                + self.difficulty_weight * difficulty_unit
            )
            ata_scores = (
                torch.log((1.0 - max_reliability).clamp_min(EPS))
                + self.beta * mean_source_entropy
                - self.difficulty_weight * mean_difficulty_unit
            )
            if self.score_mode == 'idea_calibrated':
                _assert_probability('gate_p_ata', p_ata, (reliability.size(0), None))
                _assert_probability(
                    'gate_source_mean_probs',
                    source_mean_probs,
                    (reliability.size(0), reliability.size(1), p_ata.size(1)),
                )
                ata_confidence = p_ata.detach().max(dim=1, keepdim=True).values
                source_confidence = source_mean_probs.detach().max(dim=2).values
                confidence_margin = source_confidence - ata_confidence
                _assert_shape('confidence_margin', confidence_margin, reliability.shape)
                _assert_finite('confidence_margin', confidence_margin)
                assert (confidence_margin >= -1.0 - 1e-6).all()
                assert (confidence_margin <= 1.0 + 1e-6).all()
                source_scores = source_scores + self.confidence_weight * confidence_margin
        else:
            
            
            
            difficulty_unit = difficulty / (1.0 + difficulty)
            mean_difficulty_unit = difficulty_unit.mean(dim=1, keepdim=True)
            relative_difficulty = difficulty_unit - mean_difficulty_unit
            source_scores = (
                torch.log(reliability.clamp_min(EPS))
                - self.beta * relative_difficulty
            )
            ata_scores = (
                torch.log((1.0 - max_reliability).clamp_min(EPS))
                - self.beta * mean_difficulty_unit
            )
        if abs(self.source_sharpness - 1.0) > 1e-12:
            source_center = source_scores.mean(dim=1, keepdim=True)
            source_scores = source_center + self.source_sharpness * (source_scores - source_center)

        scores = torch.cat([source_scores, ata_scores], dim=1)
        _assert_shape('moe_scores', scores, (reliability.size(0), reliability.size(1) + 1))
        _assert_finite('moe_scores', scores)
        return scores

    def forward(
        self,
        reliability,
        difficulty,
        source_entropy=None,
        p_ata=None,
        source_mean_probs=None,
    ):
        _assert_shape('reliability', reliability, (None, None))
        num_nodes, num_sources = reliability.shape
        _assert_shape('difficulty', difficulty, (num_nodes, num_sources))
        _assert_finite('reliability', reliability)
        _assert_finite('difficulty', difficulty)
        assert num_sources > 0, 'PriorMoEGate requires at least one source expert'
        assert (reliability >= -1e-6).all(), 'reliability must be non-negative'
        assert (reliability <= 1.0 + 1e-6).all(), 'reliability must not exceed 1'
        assert (difficulty >= -1e-6).all(), 'difficulty must be non-negative'

        reliability = reliability.clamp(0.0, 1.0)
        difficulty = difficulty.clamp_min(0.0)
        scores = self._build_scores(
            reliability,
            difficulty,
            source_entropy=source_entropy,
            p_ata=p_ata,
            source_mean_probs=source_mean_probs,
        )
        gate = self._normalize_scores(scores)

        gate_ata = gate[:, -1:].clamp_min(self.ata_floor)
        gate_src_sum = gate[:, :-1].sum(dim=1, keepdim=True)
        gate_src = gate[:, :-1] * (1.0 - gate_ata) / gate_src_sum.clamp_min(EPS)
        gate = torch.cat([gate_src, gate_ata], dim=1)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(EPS)

        _assert_probability('gate', gate, (num_nodes, num_sources + 1))
        assert (gate[:, -1] >= self.ata_floor - 1e-6).all(), 'ATA gate violates ata_floor'
        _debug_tensor('reliability', reliability, self.debug)
        _debug_tensor('difficulty', difficulty, self.debug)
        if source_entropy is not None:
            _debug_tensor('source_entropy', source_entropy, self.debug)
        _debug_tensor('moe_scores', scores, self.debug)
        _debug_tensor('gate', gate, self.debug)
        if self.debug:
            print('[M3] prior_gate score_mode={} ata_floor={:.6f} beta={:.6f} source_sharpness={:.6f} difficulty_weight={:.6f} confidence_weight={:.6f}'.format(
                self.score_mode,
                self.ata_floor,
                self.beta,
                self.source_sharpness,
                self.difficulty_weight,
                self.confidence_weight,
            ))
        return gate


class TrainableMoEGate(nn.Module):
    def __init__(
        self,
        num_sources,
        probe_dim=5,
        hidden=64,
        ata_floor=0.10,
        use_sparsemax=True,
        debug=False,
    ):
        super(TrainableMoEGate, self).__init__()
        assert num_sources > 0, 'TrainableMoEGate requires at least one source expert'
        assert probe_dim > 0, 'probe_dim must be positive'
        assert hidden > 0, 'hidden must be positive'
        assert 0.0 <= ata_floor <= 1.0, 'ata_floor must be in [0, 1]'
        self.num_sources = int(num_sources)
        self.probe_dim = int(probe_dim)
        self.context_dim = 6
        self.hidden = int(hidden)
        self.ata_floor = float(ata_floor)
        self.debug = debug
        self.use_sparsemax = bool(use_sparsemax)
        self.sparsemax = None
        if self.use_sparsemax:
            try:
                from layer import Sparsemax
                self.sparsemax = Sparsemax(dim=1)
            except Exception:
                self.sparsemax = None
                self.use_sparsemax = False

        input_dim = self.num_sources * (self.probe_dim + 2 + self.context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden, self.num_sources + 1),
        )

    def _normalize_scores(self, scores):
        _assert_shape('trainable_moe_scores', scores, (None, self.num_sources + 1))
        _assert_finite('trainable_moe_scores', scores)
        if self.use_sparsemax and self.sparsemax is not None:
            gate = self.sparsemax(scores)
            gate = _normalize_probability(gate)
        else:
            gate = F.softmax(scores, dim=1)
        _assert_probability('trainable_gate_before_floor', gate, (scores.size(0), self.num_sources + 1))
        return gate

    def _build_expert_context(self, p_ata, source_mean_probs, num_nodes):
        if p_ata is None or source_mean_probs is None:
            return torch.zeros(
                num_nodes,
                self.num_sources,
                self.context_dim,
                device=next(self.parameters()).device,
            )

        _assert_probability('trainable_gate_p_ata', p_ata, (num_nodes, None))
        num_classes = p_ata.size(1)
        _assert_probability(
            'trainable_gate_source_mean_probs',
            source_mean_probs,
            (num_nodes, self.num_sources, num_classes),
        )
        source_conf = source_mean_probs.max(dim=2).values
        ata_conf = p_ata.max(dim=1).values.unsqueeze(1).expand(-1, self.num_sources)
        entropy_scale = max(float(torch.log(p_ata.new_tensor(float(num_classes))).item()), EPS)
        source_entropy = -(
            source_mean_probs * torch.log(source_mean_probs.clamp_min(EPS))
        ).sum(dim=2) / entropy_scale
        ata_entropy = -(
            p_ata * torch.log(p_ata.clamp_min(EPS))
        ).sum(dim=1) / entropy_scale
        ata_entropy = ata_entropy.unsqueeze(1).expand(-1, self.num_sources)
        ata_label = p_ata.argmax(dim=1).view(num_nodes, 1, 1).expand(-1, self.num_sources, 1)
        source_at_ata = source_mean_probs.gather(2, ata_label).squeeze(2)
        source_ata_similarity = (
            source_mean_probs * p_ata.unsqueeze(1)
        ).sum(dim=2)
        confidence_advantage = source_conf - ata_conf
        context = torch.stack(
            [
                source_conf,
                source_entropy,
                source_at_ata,
                source_ata_similarity,
                ata_conf,
                ata_entropy,
            ],
            dim=2,
        )
        _assert_shape(
            'trainable_gate_context',
            context,
            (num_nodes, self.num_sources, self.context_dim),
        )
        _assert_finite('trainable_gate_context', context)
        _assert_finite('trainable_gate_confidence_advantage', confidence_advantage)
        return context.detach()

    def forward(self, probe_feat, reliability, difficulty, p_ata=None, source_mean_probs=None):
        _assert_shape('probe_feat', probe_feat, (None, self.num_sources, self.probe_dim))
        num_nodes = probe_feat.size(0)
        _assert_shape('reliability', reliability, (num_nodes, self.num_sources))
        _assert_shape('difficulty', difficulty, (num_nodes, self.num_sources))
        _assert_finite('probe_feat', probe_feat)
        _assert_finite('reliability', reliability)
        _assert_finite('difficulty', difficulty)
        assert (reliability >= -1e-6).all(), 'reliability must be non-negative'
        assert (reliability <= 1.0 + 1e-6).all(), 'reliability must not exceed 1'
        assert (difficulty >= -1e-6).all(), 'difficulty must be non-negative'

        reliability = reliability.clamp(0.0, 1.0)
        difficulty = difficulty.clamp_min(0.0)
        expert_context = self._build_expert_context(
            p_ata,
            source_mean_probs,
            num_nodes,
        ).to(device=probe_feat.device, dtype=probe_feat.dtype)
        gate_input = torch.cat(
            [
                probe_feat,
                reliability.unsqueeze(-1),
                difficulty.unsqueeze(-1),
                expert_context,
            ],
            dim=-1,
        )
        _assert_shape(
            'trainable_gate_input',
            gate_input,
            (num_nodes, self.num_sources, self.probe_dim + 2 + self.context_dim),
        )
        _assert_finite('trainable_gate_input', gate_input)
        gate_input = gate_input.reshape(
            num_nodes,
            self.num_sources * (self.probe_dim + 2 + self.context_dim),
        ).detach()
        scores = self.mlp(gate_input)
        gate = self._normalize_scores(scores)

        gate_ata = gate[:, -1:].clamp_min(self.ata_floor)
        gate_src_sum = gate[:, :-1].sum(dim=1, keepdim=True)
        gate_src = gate[:, :-1] * (1.0 - gate_ata) / gate_src_sum.clamp_min(EPS)
        gate = torch.cat([gate_src, gate_ata], dim=1)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(EPS)

        _assert_probability('trainable_gate', gate, (num_nodes, self.num_sources + 1))
        assert (gate[:, -1] >= self.ata_floor - 1e-6).all(), 'ATA gate violates ata_floor'
        _debug_tensor('trainable_gate_input', gate_input, self.debug)
        _debug_tensor('trainable_moe_scores', scores, self.debug)
        _debug_tensor('trainable_gate', gate, self.debug)
        return gate


def gate_prior_loss(prior_gate, train_gate, debug=False):
    _assert_shape('prior_gate', prior_gate, (None, None))
    num_nodes, num_experts = prior_gate.shape
    _assert_probability('prior_gate', prior_gate, (num_nodes, num_experts))
    _assert_probability('train_gate', train_gate, (num_nodes, num_experts))
    loss_gate = torch.sum(
        prior_gate.detach()
        * (
            torch.log(prior_gate.detach().clamp_min(EPS))
            - torch.log(train_gate.clamp_min(EPS))
        ),
        dim=1,
    ).mean()
    _assert_finite('loss_gate_prior', loss_gate)
    if debug:
        print('[M3] loss_gate_prior={:.6f}'.format(loss_gate.item()))
    return loss_gate


def mean_source_probs(source_probs, source_temperature=1.0, debug=False):
    """
    source_probs: [N, M, S+1, C]
    return: [N, M, C]
    """
    assert source_temperature > 0.0, 'source_temperature must be positive'
    _assert_shape('source_probs', source_probs, (None, None, None, None))
    num_nodes, num_sources, _, num_classes = source_probs.shape
    _assert_probability('source_probs', source_probs, (num_nodes, num_sources, None, num_classes))
    source_mean_probs = source_probs.detach().mean(dim=2)
    source_mean_probs = _normalize_probability(source_mean_probs)
    if abs(float(source_temperature) - 1.0) > 1e-12:
        source_mean_probs = _temperature_calibrate_probability(
            source_mean_probs,
            source_temperature,
            'source_mean_probs',
        )
    _assert_probability('source_mean_probs', source_mean_probs, (num_nodes, num_sources, num_classes))
    assert not source_mean_probs.requires_grad, 'source_mean_probs must be detached'
    _debug_tensor('source_mean_probs', source_mean_probs, debug)
    if debug:
        print('[M3] source_temperature={:.6f}'.format(float(source_temperature)))
    return source_mean_probs.detach()


def fuse_moe_predictions(
    p_ata,
    source_probs,
    gate,
    source_temperature=1.0,
    ata_temperature=1.0,
    debug=False,
):
    """
    p_ata: [N, C]
    source_probs: [N, M, S+1, C]
    gate: [N, M+1]
    return:
        final_probs: [N, C]
        source_mean_probs: [N, M, C]
    """
    assert source_temperature > 0.0, 'source_temperature must be positive'
    assert ata_temperature > 0.0, 'ata_temperature must be positive'
    _assert_shape('p_ata', p_ata, (None, None))
    num_nodes, num_classes = p_ata.shape
    _assert_probability('p_ata', p_ata, (num_nodes, num_classes))
    _assert_shape('source_probs', source_probs, (num_nodes, None, None, num_classes))
    num_sources = source_probs.size(1)
    _assert_probability('gate', gate, (num_nodes, num_sources + 1))

    source_mean = mean_source_probs(
        source_probs,
        source_temperature=source_temperature,
        debug=debug,
    )
    p_ata_for_fusion = p_ata
    if abs(float(ata_temperature) - 1.0) > 1e-12:
        p_ata_for_fusion = _temperature_calibrate_probability(
            p_ata,
            ata_temperature,
            'p_ata_for_fusion',
        )
    _assert_probability('p_ata_for_fusion', p_ata_for_fusion, (num_nodes, num_classes))
    src_part = (gate[:, :-1].unsqueeze(-1) * source_mean).sum(dim=1)
    ata_part = gate[:, -1:] * p_ata_for_fusion
    final_probs = src_part + ata_part
    final_probs = _normalize_probability(final_probs)

    _assert_probability('final_probs', final_probs, (num_nodes, num_classes))
    assert not source_mean.requires_grad, 'source_mean_probs must be detached'
    _debug_tensor('final_probs', final_probs, debug)
    if debug:
        row_sum_error = torch.max(torch.abs(final_probs.sum(dim=1) - 1.0)).item()
        print('[M3] final_probs row sum max error: {:.8f}'.format(row_sum_error))
        print('[M3] ata_temperature={:.6f}'.format(float(ata_temperature)))
    return final_probs, source_mean


def dynamic_moe_loss(
    final_probs,
    p_ata,
    pseudo_y,
    pseudo_dist,
    q_clean,
    q_noise,
    p_probe=None,
    q_diff=None,
    final_probs_strong=None,
    lambda_clean=1.0,
    lambda_noise=0.5,
    lambda_sim=0.0,
    lambda_reg=1.0,
    lambda_ata_anchor=0.2,
    debug=False,
):
    _assert_shape('final_probs', final_probs, (None, None))
    num_nodes, num_classes = final_probs.shape
    _assert_probability('final_probs', final_probs, (num_nodes, num_classes))
    _assert_probability('p_ata', p_ata, (num_nodes, num_classes))
    _assert_shape('pseudo_y', pseudo_y, (num_nodes,))
    assert pseudo_y.dtype == torch.long, 'pseudo_y must have dtype torch.long'
    assert (pseudo_y >= 0).all() and (pseudo_y < num_classes).all(), 'pseudo_y is out of range'
    _assert_probability('pseudo_dist', pseudo_dist, (num_nodes, num_classes))
    if p_probe is not None:
        _assert_probability('p_probe', p_probe, (num_nodes, num_classes))
    _assert_unit_interval('q_clean', q_clean, (num_nodes,))
    _assert_unit_interval('q_noise', q_noise, (num_nodes,))
    assert lambda_clean >= 0.0, 'lambda_clean must be non-negative'
    assert lambda_noise >= 0.0, 'lambda_noise must be non-negative'
    assert lambda_sim >= 0.0, 'lambda_sim must be non-negative'
    assert lambda_reg >= 0.0, 'lambda_reg must be non-negative'
    assert lambda_ata_anchor >= 0.0, 'lambda_ata_anchor must be non-negative'

    log_final = torch.log(final_probs.clamp_min(EPS))
    log_ata = torch.log(p_ata.clamp_min(EPS))

    clean_ce = F.nll_loss(log_final, pseudo_y, reduction='none')
    loss_clean = torch.mean(q_clean.detach() * clean_ce)

    if p_probe is not None:
        soft_target = (
            0.4 * pseudo_dist.detach()
            + 0.4 * p_probe.detach()
            + 0.2 * p_ata.detach()
        )
        soft_target_mode = 'pseudo_probe_ata'
        soft_target_mode_id = final_probs.new_tensor(1.0)
    else:
        soft_target = 0.5 * pseudo_dist.detach() + 0.5 * p_ata.detach()
        soft_target_mode = 'pseudo_ata'
        soft_target_mode_id = final_probs.new_tensor(0.0)
    soft_target = _normalize_probability(soft_target)
    _assert_probability('soft_target', soft_target, (num_nodes, num_classes))
    soft_conf = soft_target.max(dim=1).values
    soft_ce = -(soft_target.detach() * log_final).sum(dim=1)
    loss_noise = torch.mean(q_noise.detach() * soft_conf.detach() * soft_ce)

    loss_sim = final_probs.sum() * 0.0
    if q_diff is not None:
        _assert_unit_interval('q_diff', q_diff, (num_nodes,))
    if lambda_sim > 0.0 and q_diff is not None and final_probs_strong is not None:
        if final_probs_strong.dim() == 2:
            _assert_probability(
                'final_probs_strong',
                final_probs_strong,
                (num_nodes, num_classes),
            )
            sim_per_node = torch.sum((final_probs - final_probs_strong) ** 2, dim=1)
        else:
            _assert_probability(
                'final_probs_strong_all',
                final_probs_strong,
                (num_nodes, None, num_classes),
            )
            assert final_probs_strong.size(1) > 0, 'strong-view dimension must be non-empty'
            sim_per_node = torch.sum(
                (final_probs.unsqueeze(1) - final_probs_strong) ** 2,
                dim=2,
            ).mean(dim=1)
        _assert_finite('sim_per_node', sim_per_node)
        loss_sim = torch.mean(q_diff.detach() * sim_per_node)

    loss_reg = _entropy_regularization(final_probs)
    anchor_ce = F.nll_loss(log_ata, pseudo_y, reduction='none')
    loss_anchor = torch.mean(q_clean.detach() * anchor_ce)

    loss_total = (
        lambda_clean * loss_clean
        + lambda_noise * loss_noise
        + lambda_sim * loss_sim
        + lambda_reg * loss_reg
        + lambda_ata_anchor * loss_anchor
    )
    _assert_finite('loss_clean', loss_clean)
    _assert_finite('loss_noise', loss_noise)
    _assert_finite('loss_sim', loss_sim)
    _assert_finite('loss_reg', loss_reg)
    _assert_finite('loss_anchor', loss_anchor)
    _assert_finite('loss_total', loss_total)

    _debug_tensor('soft_target', soft_target, debug)
    if debug:
        print('[M3] soft_target_mode={}'.format(soft_target_mode))
        print(
            '[M3] loss_clean={:.6f} loss_noise={:.6f} loss_sim={:.6f} '
            'loss_reg={:.6f} loss_anchor={:.6f} loss_total={:.6f}'.format(
                loss_clean.item(),
                loss_noise.item(),
                loss_sim.item(),
                loss_reg.item(),
                loss_anchor.item(),
                loss_total.item(),
            )
        )

    loss_dict = {
        'loss_clean': loss_clean,
        'loss_noise': loss_noise,
        'loss_sim': loss_sim,
        'loss_reg': loss_reg,
        'loss_anchor': loss_anchor,
        'loss_total': loss_total,
        'soft_target_mode_id': soft_target_mode_id,
    }
    return loss_total, loss_dict


@torch.no_grad()
def gate_statistics(gate, debug=False, prefix='[M3]'):
    _assert_shape('gate', gate, (None, None))
    num_nodes, num_experts = gate.shape
    assert num_experts >= 2, 'gate must include at least one source and ATA expert'
    _assert_probability('gate', gate, (num_nodes, num_experts))

    row_sum_error = torch.max(torch.abs(gate.sum(dim=1) - 1.0)).item()
    source_gate = gate[:, :-1]
    ata_gate = gate[:, -1]
    stats = {
        'gate_shape': [num_nodes, num_experts],
        'gate_row_sum_max_error': row_sum_error,
        'gate_source_mean': source_gate.mean(dim=0).detach().cpu().tolist(),
        'gate_ata_mean': ata_gate.mean().item(),
        'gate_ata_min': ata_gate.min().item(),
        'gate_ata_max': ata_gate.max().item(),
        'gate_max_expert_hist': torch.bincount(
            gate.argmax(dim=1).detach().cpu(),
            minlength=num_experts,
        ).tolist(),
    }
    if debug:
        print('{} gate shape: {}'.format(prefix, stats['gate_shape']))
        print('{} gate row sum max error: {:.8f}'.format(prefix, stats['gate_row_sum_max_error']))
        print('{} gate source mean list: {}'.format(prefix, stats['gate_source_mean']))
        print(
            '{} gate ata mean/min/max: {:.6f}/{:.6f}/{:.6f}'.format(
                prefix,
                stats['gate_ata_mean'],
                stats['gate_ata_min'],
                stats['gate_ata_max'],
            )
        )
        print('{} gate max expert hist: {}'.format(prefix, stats['gate_max_expert_hist']))
    return stats


__all__ = [
    'PriorMoEGate',
    'TrainableMoEGate',
    'gate_prior_loss',
    'mean_source_probs',
    'fuse_moe_predictions',
    'dynamic_moe_loss',
    'gate_statistics',
]
