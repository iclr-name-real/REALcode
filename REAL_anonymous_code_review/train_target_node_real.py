import argparse
import os
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from utils import *
from layer import *
from datasets import *
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2080, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--lr_decay_epoch',
    type=int,
    default=-1,
    help='After this epoch, multiply optimizer LR by --lr_decay_factor; -1 disables.',
)
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:2', help='specify cuda devices')
parser.add_argument('--target', type=str, default='DBLPv7', help='target domain data')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--results_path', type=str, default='results.txt',
                    help='Path for the final CSV-style result record.')
parser.add_argument('--momentum', '--memory_gamma', dest='momentum', type=float, default=0.9, help='memory bank smoothing parameter gamma')
parser.add_argument('--lambda_tradeoff', type=float, default=0.2, help='trade-off hyperparameter lambda in [0, 1]')
parser.add_argument('--train_source_params', action='store_true', help='also update loaded source model parameters during target adaptation')
parser.add_argument('--train_source_classifiers', action='store_true', help='update only loaded source classifier heads during target adaptation')
parser.add_argument('--disable_static_cache', action='store_true', help='disable cached source projections for the static first target layer')
parser.add_argument('--uniform_attention_init', action='store_true', help='initialize source attention logits to uniform weights')
parser.add_argument('--memory_init', type=str, default='model', choices=['model', 'source_ensemble'], help='initial prediction memory source')
parser.add_argument('--pretrain_dir', type=str, default='pretrain',
                    help='Directory containing source model weights.')
parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')
parser.add_argument('--K', type=int, default=40, help='number of nearest neighbors')
parser.add_argument('--apd_m1', action='store_true',
                    help='Enable SPRE: attack-probed source response.')
parser.add_argument('--apd_num_views', type=int, default=5)
parser.add_argument('--apd_edge_drop', type=float, default=0.10)
parser.add_argument('--apd_edge_add', type=float, default=0.05)
parser.add_argument('--apd_view_policy', type=str, default='alternate',
                    choices=['alternate', 'drop_only', 'drop_add'])
parser.add_argument('--source_reliability_interval', type=int, default=5)
parser.add_argument(
    '--source_reliability_rng_mode',
    type=str,
    default='legacy',
    choices=['legacy', 'isolated'],
    help='Isolate SPRE perturbation sampling from the target-model RNG stream.',
)
parser.add_argument('--source_reliability_temp', type=float, default=1.0)
parser.add_argument('--apd_lambda_probe', type=float, default=0.5)
parser.add_argument('--apd_rnode_mode', type=str, default='max',
                    choices=['max', 'mean', 'topk_mean'])
parser.add_argument(
    '--apd_m1_m2_coupling',
    type=str,
    default='legacy',
    choices=['legacy', 'reliability_weighted', 'adaptive_target', 'full'],
    help=(
        'Optional SPRE-to-PHTA coupling after PHTA warmup. legacy preserves '
        'the established training path exactly.'
    ),
)
parser.add_argument('--apd_m2', action='store_true',
                    help='Enable PHTA: pseudo-wrong-event difficulty modeling.')
parser.add_argument('--apd_warmup_epochs', type=int, default=10)
parser.add_argument('--apd_m2_ramp_epochs', type=int, default=0,
                    help='Linearly ramp PHTA weights after warmup; 0 keeps the original switch.')
parser.add_argument('--apd_clean_floor', type=float, default=0.0,
                    help='Minimum hard pseudo-label CE weight for PHTA clean loss.')
parser.add_argument('--apd_lambda_clean', type=float, default=1.0,
                    help='Scale for the existing PHTA clean pseudo-label loss.')
parser.add_argument('--apd_lambda_noise', type=float, default=0.05)
parser.add_argument('--apd_lambda_sim', type=float, default=0.2)
parser.add_argument(
    '--apd_sim_view_mode',
    type=str,
    default='first',
    choices=['first', 'all'],
    help=(
        'Use the first perturbed view (legacy) or average difficult-node '
        'consistency over every SPRE perturbed view.'
    ),
)
parser.add_argument(
    '--apd_pwed_prior_mix',
    type=float,
    default=0.0,
    help=(
        'Mix the uniform marginal prior with the detached SPRE response prior '
        'inside the PHTA information-maximization loss. 0 preserves GraphATA.'
    ),
)
parser.add_argument('--apd_wrong_update_interval', type=int, default=1)
parser.add_argument('--apd_wrong_signal', type=str, default='confidence',
                    choices=['argmax', 'confidence', 'soft', 'kl'],
                    help='Wrong-event signal used by PHTA.')
parser.add_argument('--apd_wrong_momentum', type=float, default=0.9,
                    help='EMA momentum for PHTA wrong-event score.')
parser.add_argument('--apd_expert_wrong_signal', type=str, default='argmax',
                    choices=['argmax', 'confidence'])
parser.add_argument('--apd_expert_wrong_momentum', type=float, default=0.9)
parser.add_argument('--apd_wrong_norm', type=str, default='none',
                    choices=['none', 'robust', 'rank'])
parser.add_argument('--apd_gap_floor_mode', type=str, default='none',
                    choices=['none', 'max', 'mix'],
                    help='Add SPRE gap-based floor to PHTA difficulty.')
parser.add_argument('--apd_gap_floor_mix', type=float, default=0.4,
                    help='Gap-node mixture weight for --apd_gap_floor_mode mix.')
parser.add_argument('--apd_source_prune_mode', type=str, default='none',
                    choices=['none', 'global_top1', 'local_top1'],
                    help='Prune weak source experts in M2 reliability before M3 gating.')
parser.add_argument('--apd_source_prune_floor', type=float, default=0.0,
                    help='Reliability multiplier floor for pruned source experts.')
parser.add_argument('--apd_reliability_mode', type=str, default='legacy',
                    choices=['legacy', 'probe', 'probe_expert', 'pwed_rescue'],
                    help=(
                        'Build RDEA source reliability from the legacy pseudo-label-coupled '
                        'score, SPRE probe reliability only, or probe reliability times the '
                        'expert pseudo-label agreement score. pwed_rescue relaxes that agreement '
                        'penalty only for PHTA noisy/difficult nodes.'
                    ))
parser.add_argument('--apd_use_bmm', action='store_true')
parser.add_argument('--apd_bmm_interval', type=int, default=5)
parser.add_argument('--apd_bmm_min_count', type=int, default=20)
parser.add_argument('--apd_bmm_max_iter', type=int, default=10)
parser.add_argument('--apd_sim_detach_weak', action='store_true',
                    help='Stop gradients through weak-view probabilities in PHTA sim loss.')
parser.add_argument('--apd_m3', action='store_true',
                    help='Enable RDEA: difficulty-aware MoE gate.')
parser.add_argument('--apd_m3_mode', type=str, default='prior_gate',
                    choices=['prior_gate', 'trainable_gate', 'csrf'],
                    help='M3 mode: PriorGate, TrainableGate, or CSRF source residual fusion.')
parser.add_argument('--apd_beta', type=float, default=1.0)
parser.add_argument('--apd_ata_floor', type=float, default=0.10)
parser.add_argument('--apd_gate_score_mode', type=str, default='legacy',
                    choices=[
                        'legacy', 'difficulty_to_source', 'source_trust',
                        'idea_entropy', 'idea_joint', 'idea_calibrated', 'pwed_balanced',
                    ],
                    help='PriorGate score design. legacy preserves the previous behavior.')
parser.add_argument('--apd_gate_source_sharpness', type=float, default=1.0,
                    help='Sharpen source expert scores inside RDEA PriorGate. 1.0 preserves the old behavior.')
parser.add_argument('--apd_gate_difficulty_weight', type=float, default=1.0,
                    help='PHTA difficulty contribution in the RDEA idea_joint score.')
parser.add_argument('--apd_gate_confidence_weight', type=float, default=1.0,
                    help='Source-vs-ATA confidence margin weight in RDEA idea_calibrated.')
parser.add_argument(
    '--apd_eval_gate_sweep',
    type=str,
    default='',
    help=(
        'Eval-only PriorGate sweep. Semicolon-separated entries use '
        'name:score_mode:difficulty_weight:ata_floor[:confidence_weight]. '
        'Training is unchanged.'
    ),
)
parser.add_argument(
    '--apd_eval_gate_sweep_interval',
    type=int,
    default=1,
    help='Evaluate auxiliary PriorGate sweep entries every N epochs.',
)
parser.add_argument('--apd_lambda_ata_anchor', type=float, default=0.2)
parser.add_argument('--apd_m3_start_epoch', type=int, default=0,
                    help='Start applying M3 training loss after this zero-based epoch.')
parser.add_argument('--apd_lambda_m3', type=float, default=1.0,
                    help='Blend weight for M3 loss. 1.0 keeps the previous replacement behavior.')
parser.add_argument('--apd_m3_source_temp', type=float, default=1.0,
                    help='Temperature for frozen source probabilities before M3 fusion.')
parser.add_argument('--apd_m3_ata_temp', type=float, default=1.0,
                    help='Temperature for ATA probabilities before M3 fusion.')
parser.add_argument('--apd_eval_final', action='store_true')
parser.add_argument('--apd_eval_blend_mode', type=str, default='none',
                    choices=['none', 'source_mean', 'source_gate'])
parser.add_argument('--apd_eval_blend_gamma', type=float, default=0.0)
parser.add_argument('--apd_safe_blend', action='store_true',
                    help='Enable eval-only gated source blending without changing training loss.')
parser.add_argument('--apd_safe_blend_gamma', type=float, default=0.50)
parser.add_argument('--apd_safe_conf_margin', type=float, default=0.0)
parser.add_argument('--apd_safe_pseudo_margin', type=float, default=0.0)
parser.add_argument('--apd_safe_source_mix', type=str, default='source_mean',
                    choices=['source_mean', 'source_gate'])
parser.add_argument('--apd_m3_eval_only', action='store_true',
                    help='Use M3 only for final prediction evaluation; keep PHTA loss for training.')
parser.add_argument('--apd_m3_use_clean_weight', action='store_true',
                    help='Use clean_floor-adjusted PHTA clean weight in M3 clean and anchor losses.')
parser.add_argument('--apd_trainable_gate', action='store_true')
parser.add_argument('--apd_gate_hidden', type=int, default=64)
parser.add_argument('--apd_lambda_gate_prior', type=float, default=1.0)
parser.add_argument('--apd_memory_update_mode', type=str, default='ata',
                    choices=['ata', 'final', 'probe_final'],
                    help='Prediction used to update GraphATA memory. ata is the idea-aligned default.')
parser.add_argument('--apd_wrong_event_mode', type=str, default='final',
                    choices=['ata', 'final', 'probe_final'],
                    help='Current prediction used by PHTA wrong-event tracking.')
parser.add_argument('--apd_memory_probe_weight', type=float, default=0.5,
                    help='SPRE weight when updating the prediction memory. The remaining mass uses final MoE or ATA predictions.')
parser.add_argument('--apd_wrong_probe_weight', type=float, default=0.5,
                    help='SPRE weight in the consensus prediction used by PHTA wrong-event tracking.')
parser.add_argument('--apd_ata_mass_target', type=float, default=0.50,
                    help='Maximum desired mean GraphATA gate mass for the M3 usage penalty.')
parser.add_argument('--apd_lambda_ata_mass', type=float, default=0.10,
                    help='Weight of the M3 mean GraphATA gate-mass penalty.')
parser.add_argument('--apd_debug', action='store_true')
parser.add_argument('--apd_source_gate_diag', action='store_true',
                    help='Print source expert quality and gate prediction-change diagnostics.')
parser.add_argument(
    '--apd_best_artifact_path',
    type=str,
    default='',
    help='Optional diagnostic-only torch.save path for tensors at the best target epoch.',
)

args = parser.parse_args()


def parse_apd_eval_gate_sweep(raw_spec):
    if not raw_spec.strip():
        return []
    allowed_score_modes = {
        'legacy', 'difficulty_to_source', 'source_trust',
        'idea_entropy', 'idea_joint', 'idea_calibrated', 'pwed_balanced',
    }
    specs = []
    names = set()
    for raw_entry in raw_spec.split(';'):
        entry = raw_entry.strip()
        if not entry:
            continue
        fields = [field.strip() for field in entry.split(':')]
        if len(fields) not in (4, 5):
            raise ValueError(
                '--apd_eval_gate_sweep entries must use '
                'name:score_mode:difficulty_weight:ata_floor[:confidence_weight].'
            )
        name, score_mode, difficulty_weight_raw, ata_floor_raw = fields[:4]
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('Invalid eval gate sweep name: {}'.format(name))
        if name in names:
            raise ValueError('Duplicate eval gate sweep name: {}'.format(name))
        if score_mode not in allowed_score_modes:
            raise ValueError('Invalid eval gate sweep score mode: {}'.format(score_mode))
        difficulty_weight = float(difficulty_weight_raw)
        ata_floor = float(ata_floor_raw)
        if difficulty_weight < 0.0:
            raise ValueError('Eval gate sweep difficulty weight must be non-negative.')
        if ata_floor < 0.0 or ata_floor > 1.0:
            raise ValueError('Eval gate sweep ATA floor must be in [0, 1].')
        confidence_weight = float(fields[4]) if len(fields) == 5 else None
        if confidence_weight is not None and confidence_weight < 0.0:
            raise ValueError('Eval gate sweep confidence weight must be non-negative.')
        names.add(name)
        specs.append({
            'name': name,
            'score_mode': score_mode,
            'difficulty_weight': difficulty_weight,
            'ata_floor': ata_floor,
            'confidence_weight': confidence_weight,
        })
    return specs


apd_eval_gate_sweep_specs = parse_apd_eval_gate_sweep(args.apd_eval_gate_sweep)

_argv = sys.argv[1:]
_apd_m3_mode_explicit = any(
    item == '--apd_m3_mode' or item.startswith('--apd_m3_mode=')
    for item in _argv
)
_apd_eval_blend_mode_explicit = any(
    item == '--apd_eval_blend_mode' or item.startswith('--apd_eval_blend_mode=')
    for item in _argv
)

if not _apd_m3_mode_explicit and args.apd_trainable_gate:
    args.apd_m3_mode = 'trainable_gate'

if args.apd_m3_mode == 'csrf':
    if args.apd_trainable_gate:
        raise ValueError('--apd_m3_mode csrf cannot be combined with --apd_trainable_gate.')
    if not _apd_eval_blend_mode_explicit:
        args.apd_eval_blend_mode = 'source_mean'
    args.apd_m3_eval_only = True
elif args.apd_m3_mode == 'trainable_gate':
    if not args.apd_trainable_gate:
        raise ValueError('--apd_m3_mode trainable_gate requires --apd_trainable_gate.')
elif args.apd_m3_mode == 'prior_gate' and _apd_m3_mode_explicit and args.apd_trainable_gate:
    raise ValueError('--apd_m3_mode prior_gate cannot be combined with --apd_trainable_gate.')

if args.apd_m2 and not args.apd_m1:
    raise ValueError('--apd_m2 requires --apd_m1.')

if args.lr_decay_epoch < -1 or args.lr_decay_epoch == 0:
    raise ValueError('--lr_decay_epoch must be -1 or a positive epoch.')
if args.lr_decay_factor <= 0.0 or args.lr_decay_factor > 1.0:
    raise ValueError('--lr_decay_factor must be in (0, 1].')
if args.apd_m3 and not (args.apd_m1 and args.apd_m2):
    raise ValueError('--apd_m3 requires --apd_m1 and --apd_m2.')

if args.apd_m1:
    if args.train_source_params or args.train_source_classifiers:
        raise ValueError('--apd_m1 requires frozen source models; disable source parameter updates.')
    if args.source_reliability_interval <= 0:
        raise ValueError('--source_reliability_interval must be positive when --apd_m1 is enabled.')
    if args.apd_lambda_probe < 0.0:
        raise ValueError('--apd_lambda_probe must be non-negative.')

if args.apd_m2:
    if args.apd_warmup_epochs < 0:
        raise ValueError('--apd_warmup_epochs must be non-negative.')
    if args.apd_m2_ramp_epochs < 0:
        raise ValueError('--apd_m2_ramp_epochs must be non-negative.')
    if args.apd_clean_floor < 0.0 or args.apd_clean_floor > 1.0:
        raise ValueError('--apd_clean_floor must be in [0, 1].')
    if args.apd_lambda_clean < 0.0:
        raise ValueError('--apd_lambda_clean must be non-negative.')
    if args.apd_lambda_noise < 0.0:
        raise ValueError('--apd_lambda_noise must be non-negative.')
    if args.apd_lambda_sim < 0.0:
        raise ValueError('--apd_lambda_sim must be non-negative.')
    if args.apd_pwed_prior_mix < 0.0 or args.apd_pwed_prior_mix > 1.0:
        raise ValueError('--apd_pwed_prior_mix must be in [0, 1].')
    if args.apd_wrong_update_interval <= 0:
        raise ValueError('--apd_wrong_update_interval must be positive.')
    if args.apd_wrong_momentum < 0.0 or args.apd_wrong_momentum >= 1.0:
        raise ValueError('--apd_wrong_momentum must be in [0, 1).')
    if args.apd_expert_wrong_momentum < 0.0 or args.apd_expert_wrong_momentum >= 1.0:
        raise ValueError('--apd_expert_wrong_momentum must be in [0, 1).')
    if args.apd_gap_floor_mix < 0.0 or args.apd_gap_floor_mix > 1.0:
        raise ValueError('--apd_gap_floor_mix must be in [0, 1].')
    if args.apd_source_prune_floor < 0.0 or args.apd_source_prune_floor > 1.0:
        raise ValueError('--apd_source_prune_floor must be in [0, 1].')
    if args.apd_bmm_interval <= 0:
        raise ValueError('--apd_bmm_interval must be positive.')
    if args.apd_bmm_min_count <= 0:
        raise ValueError('--apd_bmm_min_count must be positive.')
    if args.apd_bmm_max_iter <= 0:
        raise ValueError('--apd_bmm_max_iter must be positive.')

if args.apd_m3:
    if args.apd_beta < 0.0:
        raise ValueError('--apd_beta must be non-negative.')
    if args.apd_ata_floor < 0.0 or args.apd_ata_floor > 1.0:
        raise ValueError('--apd_ata_floor must be in [0, 1].')
    if args.apd_gate_score_mode not in (
        'legacy', 'difficulty_to_source', 'source_trust',
        'idea_entropy', 'idea_joint', 'idea_calibrated', 'pwed_balanced',
    ):
        raise ValueError('--apd_gate_score_mode is invalid.')
    if args.apd_lambda_ata_anchor < 0.0:
        raise ValueError('--apd_lambda_ata_anchor must be non-negative.')
    if args.apd_m3_start_epoch < 0:
        raise ValueError('--apd_m3_start_epoch must be non-negative.')
    if args.apd_lambda_m3 < 0.0 or args.apd_lambda_m3 > 1.0:
        raise ValueError('--apd_lambda_m3 must be in [0, 1].')
    if args.apd_m3_source_temp <= 0.0:
        raise ValueError('--apd_m3_source_temp must be positive.')
    if args.apd_m3_ata_temp <= 0.0:
        raise ValueError('--apd_m3_ata_temp must be positive.')
    if args.apd_gate_source_sharpness <= 0.0:
        raise ValueError('--apd_gate_source_sharpness must be positive.')
    if args.apd_gate_difficulty_weight < 0.0:
        raise ValueError('--apd_gate_difficulty_weight must be non-negative.')
    if args.apd_gate_confidence_weight < 0.0:
        raise ValueError('--apd_gate_confidence_weight must be non-negative.')
    if args.apd_gate_hidden <= 0:
        raise ValueError('--apd_gate_hidden must be positive.')
    if args.apd_lambda_gate_prior < 0.0:
        raise ValueError('--apd_lambda_gate_prior must be non-negative.')
    if args.apd_memory_probe_weight < 0.0 or args.apd_memory_probe_weight > 1.0:
        raise ValueError('--apd_memory_probe_weight must be in [0, 1].')
    if args.apd_wrong_probe_weight < 0.0 or args.apd_wrong_probe_weight > 1.0:
        raise ValueError('--apd_wrong_probe_weight must be in [0, 1].')
    if args.apd_ata_mass_target < 0.0 or args.apd_ata_mass_target > 1.0:
        raise ValueError('--apd_ata_mass_target must be in [0, 1].')
    if args.apd_lambda_ata_mass < 0.0:
        raise ValueError('--apd_lambda_ata_mass must be non-negative.')

if apd_eval_gate_sweep_specs:
    if not (args.apd_m3 and args.apd_eval_final and args.apd_m3_eval_only):
        raise ValueError(
            '--apd_eval_gate_sweep requires --apd_m3, --apd_eval_final, '
            'and --apd_m3_eval_only.'
        )
    if args.apd_trainable_gate:
        raise ValueError('--apd_eval_gate_sweep supports PriorGate eval-only runs.')
    if args.apd_eval_gate_sweep_interval <= 0:
        raise ValueError('--apd_eval_gate_sweep_interval must be positive.')

if args.apd_eval_blend_gamma < 0.0 or args.apd_eval_blend_gamma > 1.0:
    raise ValueError('--apd_eval_blend_gamma must be in [0, 1].')

if args.apd_eval_blend_mode != 'none' and not (args.apd_m3 and args.apd_eval_final):
    raise ValueError('--apd_eval_blend_mode requires --apd_m3 and --apd_eval_final.')

if args.apd_safe_blend_gamma < 0.0 or args.apd_safe_blend_gamma > 1.0:
    raise ValueError('--apd_safe_blend_gamma must be in [0, 1].')

if args.apd_safe_conf_margin < -1.0 or args.apd_safe_conf_margin > 1.0:
    raise ValueError('--apd_safe_conf_margin must be in [-1, 1].')

if args.apd_safe_pseudo_margin < -1.0 or args.apd_safe_pseudo_margin > 1.0:
    raise ValueError('--apd_safe_pseudo_margin must be in [-1, 1].')

if args.apd_safe_blend and not (args.apd_m3 and args.apd_eval_final):
    raise ValueError('--apd_safe_blend requires --apd_m3 and --apd_eval_final.')

if args.apd_m3_eval_only and not args.apd_m3:
    raise ValueError('--apd_m3_eval_only requires --apd_m3.')

if args.apd_m3_use_clean_weight and not args.apd_m3:
    raise ValueError('--apd_m3_use_clean_weight requires --apd_m3.')

if args.apd_trainable_gate and not args.apd_m3:
    raise ValueError('--apd_trainable_gate requires --apd_m3.')

if args.apd_trainable_gate and args.apd_m3_eval_only:
    raise ValueError('--apd_trainable_gate cannot be combined with --apd_m3_eval_only.')

if args.apd_use_bmm and not args.apd_m2:
    raise ValueError('--apd_use_bmm requires --apd_m2.')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

args.cache_static_first_layer = not args.disable_static_cache

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
    names = ['DBLPv7', 'ACMv9', 'Citationv1']
elif args.target in {'DE', 'EN'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target)
    names = ['RU', 'PTBR', 'FR', 'ES']
elif args.target in {'CSBM-G4'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/CSBM', args.target)
    target_dataset = CSBMDataset(path, args.target)
    names = ['CSBM-G1', 'CSBM-G2', 'CSBM-G3']

data = target_dataset[0]

args.num_classes = len(np.unique(data.y.numpy()))

args.num_features = data.x.size(1)
args.num_nodes = data.x.size(0)

src = []
for name in names:
    if name == args.target:
        continue
    src.append(name)

args.src = src

print(args)


model_list = []
param_group = []
for i in range(len(src)):
    model = NodeClassificationModel(args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, 'model_' + src[i] + '.pth')))
    if not args.train_source_params:
        for param in model.parameters():
            param.requires_grad_(False)
        if args.train_source_classifiers:
            for param in model.gnn.cls.parameters():
                param.requires_grad_(True)
    model_list.append(model)
    if args.train_source_params:
        param_group += list(model.parameters())
    elif args.train_source_classifiers:
        param_group += list(model.gnn.cls.parameters())

weight_list = []
for model in model_list:
    w_list = []
    for name, param in model.gnn.named_parameters():
        if name[-10:] == 'lin.weight' and args.gnn == 'gcn':
            w_list.append(param)
        elif name[-12:] == 'lin_l.weight' and args.gnn == 'sage':
            w_list.append(param)
        elif name[-14:] == 'lin_src.weight' and args.gnn == 'gat':
            w_list.append(param)
        elif name[-11:] == 'nn.0.weight' and args.gnn == 'gin':
            w_list.append(param)
    weight_list.append(w_list)

weight_listv2 = list(zip(*weight_list))

model = GraphATANode(args, weight_listv2, model_list).to(args.device)
param_group += list(model.parameters())

data = data.to(args.device)

apd_build_perturbed_edge_views = None
apd_source_probe = None
apd_aggregate_source_response = None
apd_wrong_event_tracker_cls = None
apd_build_dynamic_weights = None
apd_build_reliability_difficulty = None
apd_knn_pseudo_label = None
apd_classwise_bmm_wrong_event_weights = None
apd_prior_moe_gate_cls = None
apd_trainable_moe_gate_cls = None
apd_fuse_moe_predictions = None
apd_mean_source_probs = None
apd_dynamic_moe_loss = None
apd_gate_prior_loss = None
apd_gate_statistics = None

if args.apd_m1:
    from source_reliability import (
        SourceResponseProbe,
        aggregate_source_response,
        build_perturbed_edge_views,
    )

    apd_build_perturbed_edge_views = build_perturbed_edge_views
    apd_aggregate_source_response = aggregate_source_response
    apd_source_probe = SourceResponseProbe(model_list, args.device, debug=args.apd_debug)


def apd_build_probe_edge_views(target_data, probe_step):
    view_kwargs = {
        'num_views': args.apd_num_views,
        'edge_drop': args.apd_edge_drop,
        'edge_add': args.apd_edge_add,
        'view_policy': args.apd_view_policy,
    }
    if args.source_reliability_rng_mode == 'legacy':
        return apd_build_perturbed_edge_views(target_data, **view_kwargs)

    probe_seed = (
        int(args.seed) * 1000003
        + (int(probe_step) + 2) * 9176
        + 1729
    ) % 2147483647
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cuda_devices = []
    if target_data.edge_index.is_cuda:
        cuda_devices = [target_data.edge_index.device.index]
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(probe_seed)
            np.random.seed(probe_seed)
            torch.manual_seed(probe_seed)
            if target_data.edge_index.is_cuda:
                torch.cuda.manual_seed(probe_seed)
            edge_views = apd_build_perturbed_edge_views(target_data, **view_kwargs)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
    return edge_views

if args.apd_m2:
    from prediction_history import (
        WrongEventTracker,
        build_dynamic_weights,
        build_reliability_difficulty,
        classwise_bmm_wrong_event_weights,
        knn_pseudo_label,
    )

    apd_wrong_event_tracker_cls = WrongEventTracker
    apd_build_dynamic_weights = build_dynamic_weights
    apd_build_reliability_difficulty = build_reliability_difficulty
    apd_knn_pseudo_label = knn_pseudo_label
    apd_classwise_bmm_wrong_event_weights = classwise_bmm_wrong_event_weights

if args.apd_m3:
    from expert_allocation import (
        PriorMoEGate,
        TrainableMoEGate,
        dynamic_moe_loss,
        fuse_moe_predictions,
        gate_prior_loss,
        gate_statistics,
        mean_source_probs,
    )

    apd_prior_moe_gate_cls = PriorMoEGate
    apd_trainable_moe_gate_cls = TrainableMoEGate
    apd_fuse_moe_predictions = fuse_moe_predictions
    apd_dynamic_moe_loss = dynamic_moe_loss
    apd_gate_prior_loss = gate_prior_loss
    apd_gate_statistics = gate_statistics
    apd_mean_source_probs = mean_source_probs

if args.apd_m1 and args.apd_debug:

    original_edge_index = data.edge_index.clone()
    edge_views = apd_build_probe_edge_views(data, probe_step=-1)
    assert len(edge_views) == args.apd_num_views + 1
    assert edge_views[0] is data.edge_index
    assert torch.equal(data.edge_index, original_edge_index)
    assert all(edge_view.device == data.edge_index.device for edge_view in edge_views)
    edge_sizes = [edge_view.size(1) for edge_view in edge_views]
    if args.apd_num_views > 0 and (args.apd_edge_drop > 0.0 or args.apd_edge_add > 0.0):
        changed_views = [
            edge_view.size(1) != edge_views[0].size(1)
            or not torch.equal(edge_view, edge_views[0])
            for edge_view in edge_views[1:]
        ]
        assert any(changed_views)
    else:
        changed_views = []
    print('[M1] edge view num:', len(edge_views))
    print('[M1] view_policy:', args.apd_view_policy)
    print('[M1] probe_rng_mode:', args.source_reliability_rng_mode)
    print('[M1] edge sizes:', edge_sizes)
    print('[M1] edge changed from original:', changed_views)
    print('[M1] edge devices:', [str(e.device) for e in edge_views])

optimizer_t = torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)

def apd_assert_shape(name, tensor, expected):
    assert isinstance(tensor, torch.Tensor), '{} must be a tensor'.format(name)
    assert tensor.dim() == len(expected), '{} shape mismatch: got {}, expected rank {}'.format(
        name, tuple(tensor.shape), len(expected)
    )
    for actual, exp in zip(tensor.shape, expected):
        if exp is not None:
            assert actual == exp, '{} shape mismatch: got {}, expected {}'.format(
                name, tuple(tensor.shape), expected
            )


def apd_assert_finite(name, tensor):
    if tensor.is_floating_point() or tensor.is_complex():
        assert torch.isfinite(tensor).all(), '{} contains NaN or Inf'.format(name)


def apd_assert_probability(name, tensor, expected):
    apd_assert_shape(name, tensor, expected)
    apd_assert_finite(name, tensor)
    assert (tensor >= -1e-6).all(), '{} contains negative probabilities'.format(name)
    row_sum = tensor.sum(dim=-1)
    assert torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=1e-4,
        rtol=1e-4,
    ), '{} rows do not sum to 1'.format(name)


def apd_assert_unit_interval(name, tensor, expected):
    apd_assert_shape(name, tensor, expected)
    apd_assert_finite(name, tensor)
    assert (tensor >= -1e-6).all(), '{} contains values below 0'.format(name)
    assert (tensor <= 1.0 + 1e-6).all(), '{} contains values above 1'.format(name)


def apd_debug_tensor(name, tensor, prefix='[M1]'):
    if not args.apd_debug:
        return
    stat_tensor = tensor.detach()
    if not stat_tensor.is_floating_point():
        stat_tensor = stat_tensor.float()
    print(
        '{} {} shape: {} mean={:.6f} min={:.6f} max={:.6f} std={:.6f}'.format(
            prefix,
            name,
            list(tensor.shape),
            stat_tensor.mean().item(),
            stat_tensor.min().item(),
            stat_tensor.max().item(),
            stat_tensor.std(unbiased=False).item(),
        )
    )


@torch.no_grad()
def apd_build_consensus_probs(p_ata, p_probe=None, p_final=None, probe_weight=0.5):
    """Build an APD consensus without letting GraphATA be the only teacher."""
    apd_assert_probability('consensus_p_ata', p_ata, (p_ata.size(0), args.num_classes))
    base = p_final if p_final is not None else p_ata
    apd_assert_probability('consensus_base', base, (p_ata.size(0), args.num_classes))
    if p_probe is None:
        consensus = base.detach()
    else:
        apd_assert_probability('consensus_p_probe', p_probe, (p_ata.size(0), args.num_classes))
        consensus = probe_weight * p_probe.detach() + (1.0 - probe_weight) * base.detach()
    consensus = consensus.clamp_min(0.0)
    consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(1e-12)
    apd_assert_probability('consensus_probs', consensus, (p_ata.size(0), args.num_classes))
    return consensus.detach()


@torch.no_grad()
def apd_select_reference_probs(
    p_ata,
    p_probe=None,
    p_final=None,
    mode='ata',
    probe_weight=0.5,
):
    """Select an idea-aligned ATA/final reference or the legacy APD consensus."""
    if mode not in {'ata', 'final', 'probe_final'}:
        raise ValueError('invalid APD reference mode: {}'.format(mode))
    apd_assert_probability('reference_p_ata', p_ata, (p_ata.size(0), args.num_classes))
    if mode == 'ata':
        reference = p_ata.detach()
    elif mode == 'final':
        reference = p_ata.detach() if p_final is None else p_final.detach()
    else:
        reference = apd_build_consensus_probs(
            p_ata,
            p_probe=p_probe,
            p_final=p_final,
            probe_weight=probe_weight,
        )
    apd_assert_probability('reference_probs', reference, (p_ata.size(0), args.num_classes))
    return reference.detach()


@torch.no_grad()
def apd_sharpen_memory_probs(prob):
    """Apply GraphATA-style class balancing, then restore row-wise probabilities."""
    apd_assert_probability('memory_prob_input', prob, (prob.size(0), args.num_classes))
    sharpened = prob.pow(2)
    sharpened = sharpened / sharpened.sum(dim=0, keepdim=True).clamp_min(1e-12)
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-12)
    apd_assert_probability('memory_prob_sharpened', sharpened, (prob.size(0), args.num_classes))
    return sharpened.detach()


@torch.no_grad()
def apd_evaluate_probs(prob, labels):
    apd_assert_probability('eval_prob', prob, (labels.size(0), args.num_classes))
    apd_assert_shape('eval_labels', labels, (prob.size(0),))
    assert labels.dtype == torch.long, 'eval_labels must have dtype torch.long'
    pred = prob.argmax(dim=1)
    correct = pred.eq(labels).sum().item()
    acc = correct * 1.0 / labels.size(0)
    loss = F.nll_loss(torch.log(prob.clamp_min(1e-12)), labels)
    preds = pred.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    macro_f1 = f1_score(labels_np, preds, average='macro')
    micro_f1 = f1_score(labels_np, preds, average='micro')
    return acc, macro_f1, micro_f1, loss


@torch.no_grad()
def apd_build_gate_source_entropy(source_probs):
    source_mean_probs = apd_mean_source_probs(
        source_probs,
        source_temperature=args.apd_m3_source_temp,
        debug=False,
    )
    apd_assert_probability(
        'gate_source_mean_probs',
        source_mean_probs,
        (source_probs.size(0), len(src), args.num_classes),
    )
    source_entropy = -(
        source_mean_probs * torch.log(source_mean_probs.clamp_min(1e-12))
    ).sum(dim=2)
    apd_assert_shape('gate_source_entropy', source_entropy, (source_probs.size(0), len(src)))
    apd_assert_finite('gate_source_entropy', source_entropy)
    assert (source_entropy >= -1e-6).all(), 'gate_source_entropy must be non-negative'
    apd_debug_tensor('source_entropy', source_entropy, prefix='[M3]')
    return source_entropy.detach()


@torch.no_grad()
def apd_build_eval_source_mix(source_mean_probs, gate, mix_mode):
    apd_assert_probability(
        'eval_source_mean_probs_for_mix',
        source_mean_probs,
        (None, len(src), args.num_classes),
    )
    apd_assert_probability('eval_gate_for_mix', gate, (source_mean_probs.size(0), len(src) + 1))
    assert mix_mode in ('source_mean', 'source_gate'), 'invalid eval source mix mode'

    if mix_mode == 'source_mean':
        source_mix = source_mean_probs.mean(dim=1)
    else:
        source_gate = gate[:, :-1]
        apd_assert_shape('eval_source_gate', source_gate, (source_mean_probs.size(0), len(src)))
        apd_assert_finite('eval_source_gate', source_gate)
        assert (source_gate >= -1e-6).all(), 'eval_source_gate contains negative weights'
        source_mass = source_gate.sum(dim=1, keepdim=True)
        apd_assert_shape('eval_source_mass', source_mass, (source_mean_probs.size(0), 1))
        apd_assert_finite('eval_source_mass', source_mass)
        source_gate_norm = source_gate / source_mass.clamp_min(1e-12)
        uniform_source_gate = torch.full_like(source_gate, 1.0 / float(len(src)))
        source_gate_norm = torch.where(
            source_mass > 1e-12,
            source_gate_norm,
            uniform_source_gate,
        )
        apd_assert_probability('eval_source_gate_norm', source_gate_norm, (source_mean_probs.size(0), len(src)))
        source_mix = (source_gate_norm.unsqueeze(2) * source_mean_probs).sum(dim=1)

    source_mix = source_mix.clamp_min(0.0)
    source_mix = source_mix / source_mix.sum(dim=1, keepdim=True).clamp_min(1e-12)
    apd_assert_probability('eval_source_mix', source_mix, (source_mean_probs.size(0), args.num_classes))
    return source_mix


@torch.no_grad()
def apd_safe_blend_predictions(p_ata, source_mean_probs, gate, pseudo_y, labels):
    apd_assert_probability('safe_p_ata', p_ata, (labels.size(0), args.num_classes))
    apd_assert_probability('safe_source_mean_probs', source_mean_probs, (labels.size(0), len(src), args.num_classes))
    apd_assert_probability('safe_gate', gate, (labels.size(0), len(src) + 1))
    apd_assert_shape('safe_pseudo_y', pseudo_y, (labels.size(0),))
    apd_assert_shape('safe_labels', labels, (labels.size(0),))
    assert pseudo_y.dtype == torch.long, 'safe_pseudo_y must have dtype torch.long'
    assert labels.dtype == torch.long, 'safe_labels must have dtype torch.long'
    assert pseudo_y.device == p_ata.device, 'safe_pseudo_y device mismatch'
    assert labels.device == p_ata.device, 'safe_labels device mismatch'

    source_mix = apd_build_eval_source_mix(source_mean_probs, gate, args.apd_safe_source_mix)
    source_conf = source_mix.max(dim=1).values
    ata_conf = p_ata.max(dim=1).values
    node_index = torch.arange(p_ata.size(0), device=p_ata.device)
    source_pseudo_conf = source_mix[node_index, pseudo_y]
    ata_pseudo_conf = p_ata[node_index, pseudo_y]

    apd_assert_unit_interval('safe_source_conf', source_conf, (p_ata.size(0),))
    apd_assert_unit_interval('safe_ata_conf', ata_conf, (p_ata.size(0),))
    apd_assert_unit_interval('safe_source_pseudo_conf', source_pseudo_conf, (p_ata.size(0),))
    apd_assert_unit_interval('safe_ata_pseudo_conf', ata_pseudo_conf, (p_ata.size(0),))

    allow_source = (
        source_conf >= ata_conf + args.apd_safe_conf_margin
    ) | (
        source_pseudo_conf >= ata_pseudo_conf + args.apd_safe_pseudo_margin
    )
    apd_assert_shape('safe_allow_source', allow_source, (p_ata.size(0),))

    blended = (1.0 - args.apd_safe_blend_gamma) * p_ata + args.apd_safe_blend_gamma * source_mix
    blended = blended.clamp_min(0.0)
    blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-12)
    apd_assert_probability('safe_blended_probs', blended, (p_ata.size(0), args.num_classes))

    safe_final = torch.where(allow_source.unsqueeze(1), blended, p_ata)
    safe_final = safe_final.clamp_min(0.0)
    safe_final = safe_final / safe_final.sum(dim=1, keepdim=True).clamp_min(1e-12)
    apd_assert_probability('safe_final_probs', safe_final, (p_ata.size(0), args.num_classes))

    pred_ata = p_ata.argmax(dim=1)
    pred_safe = safe_final.argmax(dim=1)
    ata_correct = pred_ata.eq(labels)
    safe_correct = pred_safe.eq(labels)
    ata_acc = ata_correct.float().mean().item()
    safe_final_acc = safe_correct.float().mean().item()
    metrics = {
        'safe_allow_rate': allow_source.float().mean().item(),
        'safe_pred_change_rate': pred_safe.ne(pred_ata).float().mean().item(),
        'safe_final_acc': safe_final_acc,
        'safe_final_minus_p_ata': safe_final_acc - ata_acc,
        'safe_fix_rate': ((~ata_correct) & safe_correct).float().mean().item(),
        'safe_break_rate': (ata_correct & (~safe_correct)).float().mean().item(),
    }
    return safe_final, source_mix, metrics


@torch.no_grad()
def apd_source_gate_diagnostics(p_ata, source_mean_probs, final_probs, gate, labels):
    apd_assert_probability('diag_p_ata', p_ata, (labels.size(0), args.num_classes))
    apd_assert_probability('diag_final_probs', final_probs, (labels.size(0), args.num_classes))
    apd_assert_probability('diag_source_mean_probs', source_mean_probs, (labels.size(0), len(src), args.num_classes))
    apd_assert_probability('diag_gate', gate, (labels.size(0), len(src) + 1))
    apd_assert_shape('diag_labels', labels, (p_ata.size(0),))
    assert labels.dtype == torch.long, 'diag_labels must have dtype torch.long'

    pred_ata = p_ata.argmax(dim=1)
    pred_final = final_probs.argmax(dim=1)
    source_pred = source_mean_probs.argmax(dim=2)
    labels_expand = labels.unsqueeze(1)
    source_correct = source_pred.eq(labels_expand)
    ata_correct = pred_ata.eq(labels)
    final_correct = pred_final.eq(labels)

    source_acc = source_correct.float().mean(dim=0)
    best_source_idx = int(source_acc.argmax().item())
    best_source_pred = source_pred[:, best_source_idx]
    best_source_correct = best_source_pred.eq(labels)
    best_source_fixable = ((~ata_correct) & best_source_correct).float().mean().item()
    best_source_break = (ata_correct & (~best_source_correct)).float().mean().item()
    best_source_change = best_source_pred.ne(pred_ata).float().mean().item()
    source_mean_prob = source_mean_probs.mean(dim=1)
    source_mean_acc, _, _, _ = apd_evaluate_probs(source_mean_prob, labels)
    source_mean_pred = source_mean_prob.argmax(dim=1)
    source_mean_change = source_mean_pred.ne(pred_ata).float().mean().item()
    source_oracle_correct = source_correct.any(dim=1)
    source_oracle_acc = source_oracle_correct.float().mean().item()
    source_oracle_fixable = ((~ata_correct) & source_oracle_correct).float().mean().item()
    expert_oracle_correct = ata_correct | source_oracle_correct
    expert_oracle_acc = expert_oracle_correct.float().mean().item()
    expert_oracle_gain = expert_oracle_acc - ata_correct.float().mean().item()

    source_gate = gate[:, :-1]
    source_mass = source_gate.sum(dim=1)
    uniform_source_gate = torch.full_like(source_gate, 1.0 / float(len(src)))
    source_gate_norm = source_gate / source_mass.unsqueeze(1).clamp_min(1e-12)
    source_gate_norm = torch.where(
        source_mass.unsqueeze(1) > 1e-12,
        source_gate_norm,
        uniform_source_gate,
    )
    source_gate_mix = (source_gate_norm.unsqueeze(2) * source_mean_probs).sum(dim=1)
    source_gate_mix = source_gate_mix / source_gate_mix.sum(dim=1, keepdim=True).clamp_min(1e-12)
    source_gate_mix_acc, _, _, _ = apd_evaluate_probs(source_gate_mix, labels)
    source_gate_mix_pred = source_gate_mix.argmax(dim=1)
    source_gate_mix_change = source_gate_mix_pred.ne(pred_ata).float().mean().item()

    selected_source_idx = source_gate.argmax(dim=1)
    selected_source_pred = source_pred.gather(1, selected_source_idx.view(-1, 1)).squeeze(1)
    selected_source_correct = selected_source_pred.eq(labels)
    selected_source_acc = selected_source_correct.float().mean().item()
    selected_source_fixable = ((~ata_correct) & selected_source_correct).float().mean().item()
    selected_source_break = (ata_correct & (~selected_source_correct)).float().mean().item()
    selected_source_change = selected_source_pred.ne(pred_ata).float().mean().item()

    gate_argmax = gate.argmax(dim=1)
    gate_non_ata_rate = gate_argmax.ne(len(src)).float().mean().item()
    pred_change_rate = pred_final.ne(pred_ata).float().mean().item()
    final_fix_rate = ((~ata_correct) & final_correct).float().mean().item()
    final_break_rate = (ata_correct & (~final_correct)).float().mean().item()
    ata_acc = ata_correct.float().mean().item()
    final_acc = final_correct.float().mean().item()

    def format_blend_metrics(source_mix):
        apd_assert_probability('diag_source_blend_mix', source_mix, (labels.size(0), args.num_classes))
        metrics = []
        for gamma in [0.05, 0.10, 0.20, 0.30, 0.50]:
            blend_probs = (1.0 - gamma) * p_ata + gamma * source_mix
            blend_probs = blend_probs.clamp_min(0.0)
            blend_probs = blend_probs / blend_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            apd_assert_probability('diag_blend_probs', blend_probs, (labels.size(0), args.num_classes))
            blend_pred = blend_probs.argmax(dim=1)
            blend_correct = blend_pred.eq(labels)
            blend_acc = blend_correct.float().mean().item()
            blend_change = blend_pred.ne(pred_ata).float().mean().item()
            blend_fix = ((~ata_correct) & blend_correct).float().mean().item()
            blend_break = (ata_correct & (~blend_correct)).float().mean().item()
            metrics.append(
                '{:.2f}:{:.6f}:{:.6f}:{:.6f}:{:.6f}:{:.6f}'.format(
                    gamma,
                    blend_acc,
                    blend_acc - ata_acc,
                    blend_change,
                    blend_fix,
                    blend_break,
                )
            )
        return ';'.join(metrics)

    source_best_prob = source_mean_probs[:, best_source_idx, :]
    apd_assert_probability('diag_source_best_prob', source_best_prob, (labels.size(0), args.num_classes))
    blend_source_mean_metrics = format_blend_metrics(source_mean_prob)
    blend_source_gate_metrics = format_blend_metrics(source_gate_mix)
    blend_source_best_metrics = format_blend_metrics(source_best_prob)

    return {
        'source_acc_list': source_acc.detach().cpu().tolist(),
        'source_best_idx': best_source_idx,
        'source_best_acc': source_acc.max().item(),
        'source_best_fixable_rate': best_source_fixable,
        'source_best_break_rate': best_source_break,
        'source_best_change_rate': best_source_change,
        'source_mean_acc': float(source_mean_acc),
        'source_mean_change_rate': source_mean_change,
        'source_gate_mix_acc': float(source_gate_mix_acc),
        'source_gate_mix_change_rate': source_gate_mix_change,
        'source_oracle_acc': source_oracle_acc,
        'source_oracle_fixable_rate': source_oracle_fixable,
        'expert_oracle_acc': expert_oracle_acc,
        'expert_oracle_gain': expert_oracle_gain,
        'selected_source_acc': selected_source_acc,
        'selected_source_fixable_rate': selected_source_fixable,
        'selected_source_break_rate': selected_source_break,
        'selected_source_change_rate': selected_source_change,
        'gate_ata_mean': gate[:, -1].mean().item(),
        'gate_source_mass_mean': source_mass.mean().item(),
        'gate_non_ata_argmax_rate': gate_non_ata_rate,
        'pred_change_rate': pred_change_rate,
        'final_fix_rate': final_fix_rate,
        'final_break_rate': final_break_rate,
        'p_ata_acc': ata_acc,
        'final_acc': final_acc,
        'final_minus_p_ata': final_acc - ata_acc,
        'blend_source_mean_metrics': blend_source_mean_metrics,
        'blend_source_gate_metrics': blend_source_gate_metrics,
        'blend_source_best_metrics': blend_source_best_metrics,
    }


def train_target(target_data):
    t = time.time()
    best_acc = 0.0
    best_epoch = 0
    best_m3_active_acc = 0.0
    best_m3_active_epoch = 0
    best_artifact = None
    apd_last_source_gate_diag = None
    apd_last_safe_blend = None

    model.eval()
    with torch.no_grad():
        mem_fea = model.feat_bottleneck(target_data.x, target_data.edge_index).detach()
        if args.memory_init == 'source_ensemble':
            source_probs = []
            for source_model in model_list:
                source_model.eval()
                source_output = source_model(target_data.x, target_data.edge_index)
                source_probs.append(F.softmax(source_output, dim=1))
            mem_cls = torch.stack(source_probs, dim=0).mean(dim=0).detach()
        else:
            cls_output = model.feat_classifier(mem_fea)
            mem_cls = F.softmax(cls_output, dim=1).detach()

    apd_cache = {
        'p_probe': None,
        'source_weight': None,
        'r_node': None,
        'source_probs': None,
        'probe_feat': None,
        'gap_score': None,
        'r_probe': None,
        'edge_views': None,
        'reliability': None,
        'difficulty': None,
        'prior_gate': None,
        'gate': None,
        'final_probs': None,
        'source_mean_probs': None,
    }
    apd_wrong_tracker = None
    if args.apd_m2:
        apd_wrong_tracker = apd_wrong_event_tracker_cls(
            target_data.num_nodes,
            len(src),
            args.device,
            debug=args.apd_debug,
            wrong_signal=args.apd_wrong_signal,
            wrong_momentum=args.apd_wrong_momentum,
            expert_wrong_signal=args.apd_expert_wrong_signal,
            expert_wrong_momentum=args.apd_expert_wrong_momentum,
        )
    expert_allocation_gate = None
    apd_prior_moe_gate = None
    apd_eval_gate_sweep_gates = {}
    apd_eval_gate_sweep_state = {}
    if args.apd_m3:
        apd_prior_moe_gate = apd_prior_moe_gate_cls(
            beta=args.apd_beta,
            ata_floor=args.apd_ata_floor,
            score_mode=args.apd_gate_score_mode,
            source_sharpness=args.apd_gate_source_sharpness,
            difficulty_weight=args.apd_gate_difficulty_weight,
            confidence_weight=args.apd_gate_confidence_weight,
            use_sparsemax=True,
            debug=args.apd_debug,
        ).to(args.device)
        if args.apd_trainable_gate:
            expert_allocation_gate = apd_trainable_moe_gate_cls(
                len(src),
                probe_dim=5,
                hidden=args.apd_gate_hidden,
                ata_floor=args.apd_ata_floor,
                use_sparsemax=True,
                debug=args.apd_debug,
            ).to(args.device)
            gate_params = [param for param in expert_allocation_gate.parameters() if param.requires_grad]
            assert len(gate_params) > 0, 'TrainableMoEGate has no trainable parameters'
            existing_param_ids = {
                id(param)
                for group in optimizer_t.param_groups
                for param in group['params']
            }
            new_gate_params = [param for param in gate_params if id(param) not in existing_param_ids]
            if len(new_gate_params) > 0:
                optimizer_t.add_param_group({'params': new_gate_params})
        else:
            expert_allocation_gate = apd_prior_moe_gate
        for sweep_spec in apd_eval_gate_sweep_specs:
            sweep_name = sweep_spec['name']
            sweep_gate = apd_prior_moe_gate_cls(
                beta=args.apd_beta,
                ata_floor=sweep_spec['ata_floor'],
                score_mode=sweep_spec['score_mode'],
                source_sharpness=args.apd_gate_source_sharpness,
                difficulty_weight=sweep_spec['difficulty_weight'],
                confidence_weight=(
                    args.apd_gate_confidence_weight
                    if sweep_spec['confidence_weight'] is None
                    else sweep_spec['confidence_weight']
                ),
                use_sparsemax=True,
                debug=False,
            ).to(args.device)
            sweep_gate.eval()
            apd_eval_gate_sweep_gates[sweep_name] = sweep_gate
            apd_eval_gate_sweep_state[sweep_name] = {
                'score_mode': sweep_spec['score_mode'],
                'difficulty_weight': sweep_spec['difficulty_weight'],
                'ata_floor': sweep_spec['ata_floor'],
                'confidence_weight': (
                    args.apd_gate_confidence_weight
                    if sweep_spec['confidence_weight'] is None
                    else sweep_spec['confidence_weight']
                ),
                'best_acc': 0.0,
                'best_epoch': 0,
                'final_acc': 0.0,
            }
    apd_last_r_node_mean = None
    apd_last_loss_probe = None
    apd_last_m2 = None
    apd_last_m3 = None
    apd_bmm_cache = {
        'q_clean': None,
        'q_noise': None,
        'q_diff': None,
        'info': None,
        'mode': 'fallback',
    }

    for epoch in range(args.epochs):
        model.train()
        if args.apd_m3:
            expert_allocation_gate.train()
            if apd_prior_moe_gate is not None:
                apd_prior_moe_gate.eval()
        optimizer_t.zero_grad()
        feat_output = model.feat_bottleneck(target_data.x, target_data.edge_index)
        cls_output = model.feat_classifier(feat_output)
        softmax_out = F.softmax(cls_output, dim=1)
        p_ata = softmax_out
        entropy_loss = torch.mean(Entropy(p_ata))
        mean_softmax = p_ata.mean(dim=0)
        div_loss = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
        im_loss = entropy_loss + div_loss
        pwed_marginal_prior = None

        feat_norm = F.normalize(feat_output, dim=1)
        mem_fea_norm = F.normalize(mem_fea, dim=1)
        distance = feat_norm @ mem_fea_norm.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
        idx_near = idx_near[:, 1:]
        pred_near = torch.mean(mem_cls[idx_near], dim=1)
        _, preds = torch.max(pred_near, dim=1)
        pseudo_dist = pred_near
        pseudo_y = preds
        cls_loss = F.cross_entropy(cls_output, pseudo_y) 

        if args.apd_m1:
            apd_assert_shape('pseudo_dist', pseudo_dist, (target_data.num_nodes, args.num_classes))
            apd_assert_finite('pseudo_dist', pseudo_dist)
            apd_assert_shape('pseudo_y', pseudo_y, (target_data.num_nodes,))
            should_probe = apd_cache['p_probe'] is None or epoch % args.source_reliability_interval == 0
            if should_probe:
                edge_views = apd_build_probe_edge_views(target_data, probe_step=epoch)
                assert len(edge_views) == args.apd_num_views + 1
                assert edge_views[0] is target_data.edge_index
                assert all(edge_view.device == target_data.edge_index.device for edge_view in edge_views)
                edge_sizes = [edge_view.size(1) for edge_view in edge_views]
                if args.apd_debug:
                    changed_views = [
                        edge_view.size(1) != edge_views[0].size(1)
                        or not torch.equal(edge_view, edge_views[0])
                        for edge_view in edge_views[1:]
                    ]
                    print('[M1] epoch {:04d} view_policy: {}'.format(epoch + 1, args.apd_view_policy))
                    print('[M1] epoch {:04d} probe edge sizes: {}'.format(epoch + 1, edge_sizes))
                    print('[M1] epoch {:04d} edge changed from original: {}'.format(epoch + 1, changed_views))

                source_probs, probe_feat, gap_score, r_probe = apd_source_probe(
                    target_data.x,
                    edge_views,
                    target_data.num_nodes,
                )
                p_probe, source_weight, r_node = apd_aggregate_source_response(
                    source_probs,
                    r_probe,
                    temperature=args.source_reliability_temp,
                    rnode_mode=args.apd_rnode_mode,
                )

                apd_assert_probability(
                    'source_probs',
                    source_probs,
                    (target_data.num_nodes, len(src), len(edge_views), args.num_classes),
                )
                apd_assert_shape('probe_feat', probe_feat, (target_data.num_nodes, len(src), 5))
                apd_assert_shape('gap_score', gap_score, (target_data.num_nodes, len(src)))
                apd_assert_shape('r_probe', r_probe, (target_data.num_nodes, len(src)))
                apd_assert_probability('p_probe', p_probe, (target_data.num_nodes, args.num_classes))
                apd_assert_probability('source_weight', source_weight, (target_data.num_nodes, len(src)))
                apd_assert_shape('r_node', r_node, (target_data.num_nodes,))
                apd_assert_finite('probe_feat', probe_feat)
                apd_assert_finite('gap_score', gap_score)
                apd_assert_finite('r_probe', r_probe)
                apd_assert_finite('r_node', r_node)
                assert not source_probs.requires_grad
                assert not p_probe.requires_grad
                assert not r_node.requires_grad
                assert (r_node >= 0.0).all() and (r_node <= 1.0 + 1e-6).all()

                apd_cache['p_probe'] = p_probe.detach()
                apd_cache['source_weight'] = source_weight.detach()
                apd_cache['r_node'] = r_node.detach()
                apd_cache['source_probs'] = source_probs.detach()
                apd_cache['probe_feat'] = probe_feat.detach()
                apd_cache['gap_score'] = gap_score.detach()
                apd_cache['r_probe'] = r_probe.detach()
                apd_cache['edge_views'] = [edge_view.detach() for edge_view in edge_views]

                apd_debug_tensor('source_probs', source_probs)
                apd_debug_tensor('probe_feat', probe_feat)
                apd_debug_tensor('gap_score', gap_score)
                apd_debug_tensor('r_probe', r_probe)
                apd_debug_tensor('p_probe', p_probe)
                apd_debug_tensor('source_weight', source_weight)
                apd_debug_tensor('r_node', r_node)
                if args.apd_debug:
                    print('[M1] rnode_mode:', args.apd_rnode_mode)
                    print('[M1] source_weight mean over sources:', source_weight.mean(dim=0).detach().cpu().tolist())

            p_probe = apd_cache['p_probe']
            r_node = apd_cache['r_node']
            source_probs_cached = apd_cache['source_probs']
            probe_feat_cached = apd_cache['probe_feat']
            gap_score_cached = apd_cache['gap_score']
            r_probe_cached = apd_cache['r_probe']
            edge_views_cached = apd_cache['edge_views']
            apd_assert_probability('p_ata', p_ata, (target_data.num_nodes, args.num_classes))
            apd_assert_probability('p_probe_cached', p_probe, (target_data.num_nodes, args.num_classes))
            apd_assert_shape('r_node_cached', r_node, (target_data.num_nodes,))
            apd_assert_finite('r_node_cached', r_node)
            apd_assert_probability(
                'source_probs_cached',
                source_probs_cached,
                (target_data.num_nodes, len(src), None, args.num_classes),
            )
            apd_assert_shape('gap_score_cached', gap_score_cached, (target_data.num_nodes, len(src)))
            apd_assert_shape('r_probe_cached', r_probe_cached, (target_data.num_nodes, len(src)))
            apd_assert_shape('probe_feat_cached', probe_feat_cached, (target_data.num_nodes, len(src), 5))
            apd_assert_finite('gap_score_cached', gap_score_cached)
            apd_assert_finite('r_probe_cached', r_probe_cached)
            apd_assert_finite('probe_feat_cached', probe_feat_cached)
            assert edge_views_cached is not None and len(edge_views_cached) >= 1
            assert all(edge_view.device == target_data.edge_index.device for edge_view in edge_views_cached)

            if args.apd_m2:
                pseudo_y, pseudo_dist = apd_knn_pseudo_label(
                    feat_output,
                    mem_fea,
                    mem_cls,
                    args.K,
                    debug=args.apd_debug,
                )
                apd_assert_shape('pseudo_y_m2', pseudo_y, (target_data.num_nodes,))
                apd_assert_probability('pseudo_dist_m2', pseudo_dist, (target_data.num_nodes, args.num_classes))

                if args.apd_pwed_prior_mix > 0.0:
                    uniform_prior = torch.full_like(
                        mean_softmax,
                        1.0 / float(args.num_classes),
                    )
                    target_prior = p_probe.detach().mean(dim=0)
                    target_prior = target_prior / target_prior.sum().clamp_min(1e-12)
                    pwed_marginal_prior = (
                        (1.0 - args.apd_pwed_prior_mix) * uniform_prior
                        + args.apd_pwed_prior_mix * target_prior
                    )
                    pwed_marginal_prior = (
                        pwed_marginal_prior
                        / pwed_marginal_prior.sum().clamp_min(1e-12)
                    ).detach()
                    apd_assert_probability(
                        'pwed_marginal_prior',
                        pwed_marginal_prior.unsqueeze(0),
                        (1, args.num_classes),
                    )
                    marginal_kl = torch.sum(
                        mean_softmax
                        * (
                            torch.log(mean_softmax.clamp_min(1e-12))
                            - torch.log(pwed_marginal_prior.clamp_min(1e-12))
                        )
                    )
                    
                    
                    div_loss = marginal_kl - mean_softmax.new_tensor(
                        float(np.log(args.num_classes))
                    )
                    apd_assert_finite('pwed_div_loss', div_loss)
                    im_loss = entropy_loss + div_loss
                    apd_assert_finite('pwed_im_loss', im_loss)

            ce_per_node = F.cross_entropy(cls_output, pseudo_y, reduction='none')
            apd_assert_shape('ce_per_node', ce_per_node, (target_data.num_nodes,))
            apd_assert_finite('ce_per_node', ce_per_node)
            log_p_ata = torch.log(p_ata.clamp_min(1e-12))
            apd_assert_probability('p_ata_for_kl', p_ata, (target_data.num_nodes, args.num_classes))
            kl_per_node = F.kl_div(log_p_ata, p_probe.detach(), reduction='none').sum(dim=1)
            apd_assert_shape('kl_per_node', kl_per_node, (target_data.num_nodes,))
            apd_assert_finite('kl_per_node', kl_per_node)
            loss_probe = torch.mean((1.0 - r_node) * kl_per_node)
            apd_last_r_node_mean = r_node.detach().mean().item()
            apd_last_loss_probe = loss_probe.detach().item()

            if args.apd_m2:
                node_wrong_rate = apd_wrong_tracker.get_node_wrong_rate()
                expert_wrong_rate = apd_wrong_tracker.get_expert_wrong_rate()
                bmm_info = {'success_classes': [], 'fallback_classes': []}
                weight_mode = 'fallback'
                if epoch < args.apd_warmup_epochs:
                    q_clean, q_noise, q_diff = apd_build_dynamic_weights(
                        node_wrong_rate,
                        force_clean=True,
                        debug=args.apd_debug,
                        wrong_norm=args.apd_wrong_norm,
                    )
                    weight_mode = 'warmup'
                elif args.apd_use_bmm:
                    refresh_bmm = (
                        apd_bmm_cache['q_clean'] is None
                        or epoch % args.apd_bmm_interval == 0
                    )
                    if refresh_bmm:
                        try:
                            q_clean, q_noise, q_diff, bmm_info = apd_classwise_bmm_wrong_event_weights(
                                node_wrong_rate,
                                pseudo_y.detach(),
                                args.num_classes,
                                min_class_count=args.apd_bmm_min_count,
                                max_iter=args.apd_bmm_max_iter,
                                debug=args.apd_debug,
                            )
                            weight_mode = 'bmm'
                            if len(bmm_info.get('success_classes', [])) == 0:
                                weight_mode = 'bmm_all_fallback'
                        except Exception as exc:
                            q_clean, q_noise, q_diff = apd_build_dynamic_weights(
                                node_wrong_rate,
                                force_clean=False,
                                debug=args.apd_debug,
                                wrong_norm=args.apd_wrong_norm,
                            )
                            bmm_info = {
                                'success_classes': [],
                                'fallback_classes': list(range(args.num_classes)),
                                'fallback_reasons': {'all': str(exc)},
                            }
                            weight_mode = 'bmm_failed_fallback'
                        else:
                            apd_bmm_cache['q_clean'] = q_clean.detach()
                            apd_bmm_cache['q_noise'] = q_noise.detach()
                            apd_bmm_cache['q_diff'] = q_diff.detach()
                            apd_bmm_cache['info'] = bmm_info
                            apd_bmm_cache['mode'] = weight_mode
                    else:
                        q_clean = apd_bmm_cache['q_clean']
                        q_noise = apd_bmm_cache['q_noise']
                        q_diff = apd_bmm_cache['q_diff']
                        bmm_info = apd_bmm_cache['info']
                        weight_mode = 'bmm_reuse'
                else:
                    q_clean, q_noise, q_diff = apd_build_dynamic_weights(
                        node_wrong_rate,
                        force_clean=False,
                        debug=args.apd_debug,
                        wrong_norm=args.apd_wrong_norm,
                    )

                ramp_value = 0.0 if epoch < args.apd_warmup_epochs else 1.0
                if epoch >= args.apd_warmup_epochs and args.apd_m2_ramp_epochs > 0:
                    ramp_step = epoch - args.apd_warmup_epochs + 1
                    ramp_value = min(1.0, float(ramp_step) / float(args.apd_m2_ramp_epochs))
                    ramp_tensor = torch.as_tensor(ramp_value, device=q_clean.device, dtype=q_clean.dtype)
                    q_clean = ((1.0 - ramp_tensor) * torch.ones_like(q_clean) + ramp_tensor * q_clean).detach()
                    q_noise = (ramp_tensor * q_noise).detach()
                    q_diff = (ramp_tensor * q_diff).detach()
                    if weight_mode != 'warmup':
                        weight_mode = weight_mode + '_ramp'

                reliability, difficulty = apd_build_reliability_difficulty(
                    r_probe_cached,
                    gap_score_cached,
                    q_clean,
                    q_diff,
                    expert_wrong_rate=expert_wrong_rate,
                    reliability_mode=args.apd_reliability_mode,
                    gap_floor_mode=args.apd_gap_floor_mode,
                    gap_floor_mix=args.apd_gap_floor_mix,
                    source_prune_mode=args.apd_source_prune_mode,
                    source_prune_floor=args.apd_source_prune_floor,
                    debug=args.apd_debug,
                )
                apd_cache['reliability'] = reliability.detach()
                apd_cache['difficulty'] = difficulty.detach()

                apd_assert_unit_interval('node_wrong_rate', node_wrong_rate, (target_data.num_nodes,))
                apd_assert_unit_interval('expert_wrong_rate', expert_wrong_rate, (target_data.num_nodes, len(src)))
                apd_assert_unit_interval('q_clean', q_clean, (target_data.num_nodes,))
                apd_assert_unit_interval('q_noise', q_noise, (target_data.num_nodes,))
                apd_assert_unit_interval('q_diff', q_diff, (target_data.num_nodes,))
                apd_assert_shape('reliability', reliability, (target_data.num_nodes, len(src)))
                apd_assert_shape('difficulty', difficulty, (target_data.num_nodes, len(src)))
                apd_assert_finite('reliability', reliability)
                apd_assert_finite('difficulty', difficulty)

                p_probe_detached = p_probe.detach()
                pseudo_dist_detached = pseudo_dist.detach()
                r_node_detached = r_node.detach()
                apd_assert_unit_interval(
                    'r_node_for_m2',
                    r_node_detached,
                    (target_data.num_nodes,),
                )
                coupling_active = (
                    args.apd_m1_m2_coupling != 'legacy'
                    and epoch >= args.apd_warmup_epochs
                )
                adaptive_target_active = (
                    coupling_active
                    and args.apd_m1_m2_coupling in ('adaptive_target', 'full')
                )
                reliability_weight_active = (
                    coupling_active
                    and args.apd_m1_m2_coupling in ('reliability_weighted', 'full')
                )
                if adaptive_target_active:
                    target_reliability = r_node_detached.unsqueeze(1)
                    apd_assert_shape(
                        'target_reliability',
                        target_reliability,
                        (target_data.num_nodes, 1),
                    )
                    soft_target = (
                        target_reliability * pseudo_dist_detached
                        + (1.0 - target_reliability) * p_probe_detached
                    )
                    soft_target_mode = 'apsr_adaptive'
                else:
                    soft_target = 0.5 * pseudo_dist_detached + 0.5 * p_probe_detached
                    soft_target_mode = 'legacy_equal'
                soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
                apd_assert_probability('soft_target', soft_target, (target_data.num_nodes, args.num_classes))
                soft_target_conf = soft_target.max(dim=1).values
                apd_assert_unit_interval('soft_target_conf', soft_target_conf, (target_data.num_nodes,))

                soft_ce_per_node = -(soft_target.detach() * log_p_ata).sum(dim=1)
                apd_assert_shape('soft_ce_per_node', soft_ce_per_node, (target_data.num_nodes,))
                apd_assert_finite('soft_ce_per_node', soft_ce_per_node)

                if args.apd_clean_floor > 0.0:
                    clean_ce_weight = (
                        args.apd_clean_floor
                        + (1.0 - args.apd_clean_floor) * q_clean
                    ).detach()
                else:
                    clean_ce_weight = q_clean.detach()
                noise_ce_weight = q_noise.detach()
                if reliability_weight_active:
                    clean_ce_weight = (clean_ce_weight * r_node_detached).detach()
                    noise_ce_weight = (noise_ce_weight * (1.0 - r_node_detached)).detach()
                apd_assert_unit_interval('clean_ce_weight', clean_ce_weight, (target_data.num_nodes,))
                apd_assert_unit_interval('noise_ce_weight', noise_ce_weight, (target_data.num_nodes,))

                loss_clean = torch.mean(clean_ce_weight * ce_per_node)
                loss_noise = torch.mean(noise_ce_weight * soft_target_conf * soft_ce_per_node)
                p_ata_strong = None
                p_ata_strong_views = []
                sim_view_indices = []
                loss_sim = cls_output.sum() * 0.0
                if (
                    args.apd_lambda_sim > 0.0
                    and len(edge_views_cached) > 1
                    and q_diff.detach().max().item() > 0.0
                ):
                    weak_sim_target = p_ata.detach() if args.apd_sim_detach_weak else p_ata
                    apd_assert_probability(
                        'weak_sim_target',
                        weak_sim_target,
                        (target_data.num_nodes, args.num_classes),
                    )
                    if args.apd_sim_view_mode == 'first':
                        sim_view_indices = [1]
                    else:
                        sim_view_indices = list(range(1, len(edge_views_cached)))
                    sim_per_view = []
                    for sim_view_index in sim_view_indices:
                        strong_edge_index = edge_views_cached[sim_view_index]
                        apd_assert_shape(
                            'strong_edge_index_{}'.format(sim_view_index),
                            strong_edge_index,
                            (2, None),
                        )
                        strong_feat_output = model.feat_bottleneck(
                            target_data.x,
                            strong_edge_index,
                        )
                        strong_cls_output = model.feat_classifier(strong_feat_output)
                        strong_prob = F.softmax(strong_cls_output, dim=1)
                        apd_assert_probability(
                            'p_ata_strong_{}'.format(sim_view_index),
                            strong_prob,
                            (target_data.num_nodes, args.num_classes),
                        )
                        p_ata_strong_views.append(strong_prob)
                        sim_per_view.append(
                            torch.sum((weak_sim_target - strong_prob) ** 2, dim=1)
                        )
                    assert len(p_ata_strong_views) == len(sim_view_indices)
                    assert p_ata_strong_views, 'at least one strong view is required'
                    p_ata_strong = p_ata_strong_views[0]
                    sim_per_node = torch.stack(sim_per_view, dim=1).mean(dim=1)
                    apd_assert_shape('sim_per_node', sim_per_node, (target_data.num_nodes,))
                    apd_assert_finite('sim_per_node', sim_per_node)
                    loss_sim = torch.mean(q_diff * sim_per_node)

                apd_assert_finite('loss_clean', loss_clean)
                apd_assert_finite('loss_noise', loss_noise)
                apd_assert_finite('loss_sim', loss_sim)
                loss = (
                    im_loss
                    + args.apd_lambda_clean * loss_clean
                    + args.apd_lambda_noise * loss_noise
                    + args.apd_lambda_sim * loss_sim
                )
                apd_assert_finite('loss_m2', loss)

                if args.apd_m3:
                    loss_m2_base = loss
                    m3_train_active = epoch >= args.apd_m3_start_epoch
                    loss_gate_prior = cls_output.sum() * 0.0
                    source_entropy_for_gate = None
                    if args.apd_gate_score_mode in (
                        'idea_entropy', 'idea_joint', 'idea_calibrated'
                    ):
                        source_entropy_for_gate = apd_build_gate_source_entropy(source_probs_cached)
                    source_mean_for_prior = None
                    if args.apd_gate_score_mode == 'idea_calibrated':
                        source_mean_for_prior = apd_mean_source_probs(
                            source_probs_cached,
                            source_temperature=args.apd_m3_source_temp,
                            debug=False,
                        )
                    if args.apd_trainable_gate:
                        source_mean_for_gate = apd_mean_source_probs(
                            source_probs_cached,
                            source_temperature=args.apd_m3_source_temp,
                            debug=False,
                        )
                        with torch.no_grad():
                            prior_gate = apd_prior_moe_gate(
                                reliability.detach(),
                                difficulty.detach(),
                                source_entropy=source_entropy_for_gate,
                                p_ata=p_ata.detach(),
                                source_mean_probs=source_mean_for_gate.detach(),
                            )
                        gate = expert_allocation_gate(
                            probe_feat_cached,
                            reliability,
                            difficulty,
                            p_ata=p_ata,
                            source_mean_probs=source_mean_for_gate,
                        )
                        loss_gate_prior = apd_gate_prior_loss(
                            prior_gate,
                            gate,
                            debug=args.apd_debug,
                        )
                        apd_assert_probability('prior_gate', prior_gate, (target_data.num_nodes, len(src) + 1))
                        apd_assert_finite('loss_gate_prior', loss_gate_prior)
                    else:
                        prior_gate = None
                        gate = expert_allocation_gate(
                            reliability,
                            difficulty,
                            source_entropy=source_entropy_for_gate,
                            p_ata=p_ata,
                            source_mean_probs=source_mean_for_prior,
                        )
                    final_probs, source_mean_probs = apd_fuse_moe_predictions(
                        p_ata,
                        source_probs_cached,
                        gate,
                        source_temperature=args.apd_m3_source_temp,
                        ata_temperature=args.apd_m3_ata_temp,
                        debug=args.apd_debug,
                    )
                    apd_assert_probability('gate', gate, (target_data.num_nodes, len(src) + 1))
                    apd_assert_probability('source_mean_probs', source_mean_probs, (target_data.num_nodes, len(src), args.num_classes))
                    apd_assert_probability('final_probs', final_probs, (target_data.num_nodes, args.num_classes))
                    assert not source_mean_probs.requires_grad

                    moe_clean_weight = clean_ce_weight if args.apd_m3_use_clean_weight else q_clean
                    apd_assert_unit_interval('moe_clean_weight', moe_clean_weight, (target_data.num_nodes,))
                    m3_p_probe = apd_cache['p_probe'] if apd_cache.get('p_probe') is not None else None
                    if m3_p_probe is not None:
                        apd_assert_probability('m3_p_probe', m3_p_probe, (target_data.num_nodes, args.num_classes))
                    final_probs_strong = None
                    if (
                        not args.apd_m3_eval_only
                        and args.apd_lambda_sim > 0.0
                        and len(edge_views_cached) > 1
                        and q_diff.detach().max().item() > 0.0
                    ):
                        assert p_ata_strong_views
                        assert len(p_ata_strong_views) == len(sim_view_indices)
                        final_probs_strong_views = []
                        for sim_view_index, strong_prob in zip(
                            sim_view_indices,
                            p_ata_strong_views,
                        ):
                            strong_source_probs = source_probs_cached[
                                :, :, sim_view_index:sim_view_index + 1, :
                            ]
                            strong_final, _ = apd_fuse_moe_predictions(
                                strong_prob,
                                strong_source_probs,
                                gate,
                                source_temperature=args.apd_m3_source_temp,
                                ata_temperature=args.apd_m3_ata_temp,
                                debug=False,
                            )
                            final_probs_strong_views.append(strong_final)
                        if args.apd_sim_view_mode == 'first':
                            final_probs_strong = final_probs_strong_views[0]
                            apd_assert_probability(
                                'final_probs_strong',
                                final_probs_strong,
                                (target_data.num_nodes, args.num_classes),
                            )
                        else:
                            final_probs_strong = torch.stack(
                                final_probs_strong_views,
                                dim=1,
                            )
                            apd_assert_probability(
                                'final_probs_strong_all',
                                final_probs_strong,
                                (
                                    target_data.num_nodes,
                                    len(final_probs_strong_views),
                                    args.num_classes,
                                ),
                            )

                    loss_ata_mass = torch.relu(
                        gate[:, -1].mean() - args.apd_ata_mass_target
                    ).pow(2)
                    apd_assert_finite('loss_ata_mass', loss_ata_mass)

                    if args.apd_m3_eval_only:
                        with torch.no_grad():
                            moe_loss_total, moe_loss_dict = apd_dynamic_moe_loss(
                                final_probs=final_probs.detach(),
                                p_ata=p_ata.detach(),
                                pseudo_y=pseudo_y.detach(),
                                pseudo_dist=pseudo_dist.detach(),
                                p_probe=m3_p_probe,
                                q_clean=moe_clean_weight.detach(),
                                q_noise=noise_ce_weight.detach(),
                                q_diff=q_diff.detach(),
                                final_probs_strong=None,
                                lambda_clean=args.apd_lambda_clean,
                                lambda_noise=args.apd_lambda_noise,
                                lambda_sim=0.0,
                                lambda_reg=1.0,
                                lambda_ata_anchor=args.apd_lambda_ata_anchor,
                                debug=args.apd_debug,
                            )
                    else:
                        m3_loss_raw, moe_loss_dict = apd_dynamic_moe_loss(
                            final_probs=final_probs,
                            p_ata=p_ata,
                            pseudo_y=pseudo_y,
                            pseudo_dist=pseudo_dist,
                            p_probe=m3_p_probe,
                            q_clean=moe_clean_weight,
                            q_noise=noise_ce_weight,
                            q_diff=q_diff,
                            final_probs_strong=final_probs_strong,
                            lambda_clean=args.apd_lambda_clean,
                            lambda_noise=args.apd_lambda_noise,
                            lambda_sim=args.apd_lambda_sim,
                            lambda_reg=1.0,
                            lambda_ata_anchor=args.apd_lambda_ata_anchor,
                            debug=args.apd_debug,
                        )
                        m3_loss_train = (
                            m3_loss_raw
                            + args.apd_lambda_ata_mass * loss_ata_mass
                        )
                        if args.apd_trainable_gate:
                            m3_loss_train = m3_loss_train + args.apd_lambda_gate_prior * loss_gate_prior
                        if m3_train_active:
                            loss = (
                                (1.0 - args.apd_lambda_m3) * loss_m2_base
                                + args.apd_lambda_m3 * m3_loss_train
                            )
                        else:
                            loss = loss_m2_base
                        moe_loss_total = m3_loss_train
                    apd_assert_finite('loss_m3', loss)
                    gate_stats = apd_gate_statistics(gate, debug=args.apd_debug)
                    if prior_gate is not None:
                        prior_gate_stats = apd_gate_statistics(prior_gate, debug=args.apd_debug, prefix='[M3-prior]')
                    else:
                        prior_gate_stats = None
                    apd_cache['prior_gate'] = None if prior_gate is None else prior_gate.detach()
                    apd_cache['gate'] = gate.detach()
                    apd_cache['final_probs'] = final_probs.detach()
                    apd_cache['source_mean_probs'] = source_mean_probs.detach()
                    apd_last_m3 = {
                        'gate_ata_mean': gate_stats['gate_ata_mean'],
                        'gate_ata_min': gate_stats['gate_ata_min'],
                        'gate_ata_max': gate_stats['gate_ata_max'],
                        'gate_source_mean': gate_stats['gate_source_mean'],
                        'loss_clean': moe_loss_dict['loss_clean'].detach().item(),
                        'loss_noise': moe_loss_dict['loss_noise'].detach().item(),
                        'loss_sim': moe_loss_dict['loss_sim'].detach().item(),
                        'loss_reg': moe_loss_dict['loss_reg'].detach().item(),
                        'loss_anchor': moe_loss_dict['loss_anchor'].detach().item(),
                        'loss_gate_prior': loss_gate_prior.detach().item(),
                        'loss_ata_mass': loss_ata_mass.detach().item(),
                        'ata_mass_target': args.apd_ata_mass_target,
                        'lambda_ata_mass': args.apd_lambda_ata_mass,
                        'loss_total': moe_loss_total.detach().item(),
                        'loss_m2_base': loss_m2_base.detach().item(),
                        'loss_train': loss.detach().item(),
                        'm3_mode': args.apd_m3_mode,
                        'gate_type': 'trainable' if args.apd_trainable_gate else 'prior',
                        'gate_score_mode': args.apd_gate_score_mode,
                        'gate_source_sharpness': args.apd_gate_source_sharpness,
                        'gate_difficulty_weight': args.apd_gate_difficulty_weight,
                        'gate_confidence_weight': args.apd_gate_confidence_weight,
                        'train_mode': 'eval_only' if args.apd_m3_eval_only else ('m3_loss' if m3_train_active else 'm2_until_m3_start'),
                        'm3_train_active': bool(m3_train_active),
                        'm3_start_epoch': args.apd_m3_start_epoch,
                        'lambda_m3': args.apd_lambda_m3,
                        'source_temp': args.apd_m3_source_temp,
                        'ata_temp': args.apd_m3_ata_temp,
                        'clean_weight_mode': 'clean_floor' if args.apd_m3_use_clean_weight else 'q_clean',
                        'eval_blend_mode': args.apd_eval_blend_mode,
                        'eval_blend_gamma': args.apd_eval_blend_gamma,
                        'safe_blend': bool(args.apd_safe_blend),
                        'safe_blend_gamma': args.apd_safe_blend_gamma,
                        'safe_conf_margin': args.apd_safe_conf_margin,
                        'safe_pseudo_margin': args.apd_safe_pseudo_margin,
                        'safe_source_mix': args.apd_safe_source_mix,
                    }
                    if prior_gate_stats is not None:
                        apd_last_m3['prior_gate_ata_mean'] = prior_gate_stats['gate_ata_mean']
                        apd_last_m3['prior_gate_ata_min'] = prior_gate_stats['gate_ata_min']
                        apd_last_m3['prior_gate_ata_max'] = prior_gate_stats['gate_ata_max']

                apd_last_m2 = {
                    'node_wrong_rate_mean': node_wrong_rate.detach().mean().item(),
                    'node_wrong_rate_std': node_wrong_rate.detach().std(unbiased=False).item(),
                    'q_clean_mean': q_clean.detach().mean().item(),
                    'q_noise_mean': q_noise.detach().mean().item(),
                    'q_diff_mean': q_diff.detach().mean().item(),
                    'difficulty_mean': difficulty.detach().mean().item(),
                    'difficulty_std': difficulty.detach().std(unbiased=False).item(),
                    'loss_clean': loss_clean.detach().item(),
                    'lambda_clean': args.apd_lambda_clean,
                    'loss_noise': loss_noise.detach().item(),
                    'loss_sim': loss_sim.detach().item(),
                    'sim_view_mode': args.apd_sim_view_mode,
                    'sim_view_count': len(sim_view_indices),
                    'weight_mode': weight_mode,
                    'm1_m2_coupling': args.apd_m1_m2_coupling,
                    'coupling_active': bool(coupling_active),
                    'soft_target_mode': soft_target_mode,
                    'source_prune_mode': args.apd_source_prune_mode,
                    'source_prune_floor': args.apd_source_prune_floor,
                    'pwed_prior_mix': args.apd_pwed_prior_mix,
                    'marginal_prior_l1_uniform': (
                        0.0
                        if pwed_marginal_prior is None
                        else torch.sum(
                            torch.abs(
                                pwed_marginal_prior
                                - torch.full_like(
                                    pwed_marginal_prior,
                                    1.0 / float(args.num_classes),
                                )
                            )
                        ).item()
                    ),
                }

                apd_debug_tensor('clean_ce_weight', clean_ce_weight, prefix='[M2]')
                apd_debug_tensor('noise_ce_weight', noise_ce_weight, prefix='[M2]')
                apd_debug_tensor('node_wrong_rate', node_wrong_rate, prefix='[M2]')
                apd_debug_tensor('q_clean', q_clean, prefix='[M2]')
                apd_debug_tensor('q_noise', q_noise, prefix='[M2]')
                apd_debug_tensor('q_diff', q_diff, prefix='[M2]')
                apd_debug_tensor('expert_wrong_rate', expert_wrong_rate, prefix='[M2]')
                apd_debug_tensor('reliability', reliability, prefix='[M2]')
                apd_debug_tensor('difficulty', difficulty, prefix='[M2]')
                apd_debug_tensor('soft_target', soft_target, prefix='[M2]')
                if pwed_marginal_prior is not None:
                    apd_debug_tensor(
                        'pwed_marginal_prior',
                        pwed_marginal_prior,
                        prefix='[M2]',
                    )
                if args.apd_debug:
                    print(
                        '[M2] weight_mode={} bmm_success_classes={} bmm_fallback_classes={}'.format(
                            weight_mode,
                            bmm_info.get('success_classes', []),
                            bmm_info.get('fallback_classes', []),
                        )
                    )
                    print('[M2] wrong_norm={}'.format(args.apd_wrong_norm))
                    print('[M2] source_prune_mode={} source_prune_floor={:.6f}'.format(
                        args.apd_source_prune_mode,
                        args.apd_source_prune_floor,
                    ))
                    print(
                        '[M2] loss_clean={:.6f} loss_noise={:.6f} loss_sim={:.6f} loss_reg={:.6f}'.format(
                            loss_clean.item(),
                            loss_noise.item(),
                            loss_sim.item(),
                            im_loss.item(),
                        )
                    )
                    print('[M2] ramp_value={:.6f} clean_floor={:.6f} sim_detach_weak={} sim_view_mode={} sim_view_count={}'.format(
                        ramp_value,
                        args.apd_clean_floor,
                        args.apd_sim_detach_weak,
                        args.apd_sim_view_mode,
                        len(sim_view_indices),
                    ))
            else:
                loss_pl = torch.mean(r_node * ce_per_node)
                apd_assert_finite('loss_pl', loss_pl)
                apd_assert_finite('loss_probe', loss_probe)
                loss = im_loss + loss_pl + args.apd_lambda_probe * loss_probe

                if args.apd_debug:
                    print(
                        '[M1] loss_pl={:.6f} loss_probe={:.6f} loss_reg={:.6f}'.format(
                            loss_pl.item(),
                            loss_probe.item(),
                            im_loss.item(),
                        )
                    )
        else:
            loss = im_loss + cls_loss

        loss.backward()
        if args.apd_m1 and args.apd_debug:
            source_grad_is_none = all(
                param.grad is None
                for source_model in model_list
                for param in source_model.parameters()
            )
            target_grad_exists = any(
                param.grad is not None
                for param in model.parameters()
                if param.requires_grad
            )
            print('[M1] frozen source models grad is None:', source_grad_is_none)
            print('[M1] GraphATA target model grad exists:', target_grad_exists)
            assert source_grad_is_none
            assert target_grad_exists
            if args.apd_m3:
                gate_trainable_params = sum(
                    param.numel()
                    for param in expert_allocation_gate.parameters()
                    if param.requires_grad
                )
                print('[M3] {} gate trainable parameter count: {}'.format(
                    'trainable' if args.apd_trainable_gate else 'prior',
                    gate_trainable_params,
                ))
                if args.apd_trainable_gate:
                    assert gate_trainable_params > 0
                    gate_grad_exists = any(
                        param.grad is not None
                        for param in expert_allocation_gate.parameters()
                        if param.requires_grad
                    )
                    print('[M3] trainable gate grad exists:', gate_grad_exists)
                    if m3_train_active:
                        assert gate_grad_exists
                else:
                    assert gate_trainable_params == 0
        optimizer_t.step()
        if args.lr_decay_epoch > 0 and epoch + 1 == args.lr_decay_epoch:
            old_lr = float(optimizer_t.param_groups[0]['lr'])
            for param_group_item in optimizer_t.param_groups:
                param_group_item['lr'] = (
                    float(param_group_item['lr']) * args.lr_decay_factor
                )
            new_lr = float(optimizer_t.param_groups[0]['lr'])
            print(
                'APD_LR_EVENT epoch={} old_lr={:.10f} new_lr={:.10f} factor={:.6f}'.format(
                    epoch + 1,
                    old_lr,
                    new_lr,
                    args.lr_decay_factor,
                )
            )

        if args.apd_m2:
            should_update_wrong = (
                epoch >= args.apd_warmup_epochs
                and epoch % args.apd_wrong_update_interval == 0
            )
            if should_update_wrong:
                final_for_wrong = None
                if args.apd_m3 and epoch >= args.apd_m3_start_epoch:
                    final_for_wrong = apd_cache.get('final_probs')
                wrong_event_probs = apd_select_reference_probs(
                    p_ata.detach(),
                    p_probe=apd_cache.get('p_probe'),
                    p_final=final_for_wrong,
                    mode=args.apd_wrong_event_mode,
                    probe_weight=args.apd_wrong_probe_weight,
                )
                apd_assert_probability(
                    'wrong_event_probs',
                    wrong_event_probs,
                    (target_data.num_nodes, args.num_classes),
                )
                apd_wrong_tracker.update_node(
                    wrong_event_probs,
                    pseudo_y.detach(),
                    pseudo_dist=pseudo_dist.detach(),
                )
                apd_wrong_tracker.update_expert(apd_cache['source_probs'].detach(), pseudo_y.detach())

        model.eval()
        if args.apd_m3:
            expert_allocation_gate.eval()
        with torch.no_grad():
            feat_output = model.feat_bottleneck(target_data.x, target_data.edge_index)
            cls_output = model.feat_classifier(feat_output)
            softmax_out = F.softmax(cls_output, dim=1)
            p_ata = softmax_out
            test_acc_final = None
            test_acc_p_ata = None
            if args.apd_m3 and args.apd_eval_final:
                assert apd_cache['reliability'] is not None
                assert apd_cache['difficulty'] is not None
                assert apd_cache['source_probs'] is not None
                m3_eval_active = args.apd_m3_eval_only or epoch >= args.apd_m3_start_epoch
                if not m3_eval_active:
                    eval_gate = p_ata.new_zeros(target_data.num_nodes, len(src) + 1)
                    eval_gate[:, -1] = 1.0
                    eval_source_mean_probs = apd_mean_source_probs(
                        apd_cache['source_probs'],
                        source_temperature=args.apd_m3_source_temp,
                        debug=False,
                    )
                    eval_final_probs = p_ata
                elif args.apd_trainable_gate:
                    assert apd_cache['probe_feat'] is not None
                    apd_assert_shape('eval_probe_feat', apd_cache['probe_feat'], (target_data.num_nodes, len(src), 5))
                    eval_source_mean_for_gate = apd_mean_source_probs(
                        apd_cache['source_probs'],
                        source_temperature=args.apd_m3_source_temp,
                        debug=False,
                    )
                    eval_gate = expert_allocation_gate(
                        apd_cache['probe_feat'],
                        apd_cache['reliability'],
                        apd_cache['difficulty'],
                        p_ata=p_ata,
                        source_mean_probs=eval_source_mean_for_gate,
                    )
                else:
                    eval_source_entropy = None
                    if args.apd_gate_score_mode in (
                        'idea_entropy', 'idea_joint', 'idea_calibrated'
                    ):
                        eval_source_entropy = apd_build_gate_source_entropy(apd_cache['source_probs'])
                    eval_source_mean_for_prior = None
                    if args.apd_gate_score_mode == 'idea_calibrated':
                        eval_source_mean_for_prior = apd_mean_source_probs(
                            apd_cache['source_probs'],
                            source_temperature=args.apd_m3_source_temp,
                            debug=False,
                        )
                    eval_gate = expert_allocation_gate(
                        apd_cache['reliability'],
                        apd_cache['difficulty'],
                        source_entropy=eval_source_entropy,
                        p_ata=p_ata,
                        source_mean_probs=eval_source_mean_for_prior,
                    )
                if m3_eval_active:
                    eval_final_probs, eval_source_mean_probs = apd_fuse_moe_predictions(
                        p_ata,
                        apd_cache['source_probs'],
                        eval_gate,
                        source_temperature=args.apd_m3_source_temp,
                        ata_temperature=args.apd_m3_ata_temp,
                        debug=False,
                    )
                apd_assert_probability('eval_gate', eval_gate, (target_data.num_nodes, len(src) + 1))
                apd_assert_probability(
                    'eval_source_mean_probs',
                    eval_source_mean_probs,
                    (target_data.num_nodes, len(src), args.num_classes),
                )
                if m3_eval_active and args.apd_safe_blend:
                    eval_final_probs, eval_source_mix, apd_last_safe_blend = apd_safe_blend_predictions(
                        p_ata,
                        eval_source_mean_probs,
                        eval_gate,
                        pseudo_y.detach(),
                        target_data.y,
                    )
                elif m3_eval_active and args.apd_eval_blend_mode != 'none':
                    eval_source_mix = apd_build_eval_source_mix(
                        eval_source_mean_probs,
                        eval_gate,
                        args.apd_eval_blend_mode,
                    )
                    eval_final_probs = (
                        (1.0 - args.apd_eval_blend_gamma) * p_ata
                        + args.apd_eval_blend_gamma * eval_source_mix
                    )
                    eval_final_probs = eval_final_probs.clamp_min(0.0)
                    eval_final_probs = (
                        eval_final_probs
                        / eval_final_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    )
                apd_assert_probability('eval_final_probs', eval_final_probs, (target_data.num_nodes, args.num_classes))
                test_acc_p_ata, _, _, _ = apd_evaluate_probs(p_ata, target_data.y)
                test_acc_final, _, _, _ = apd_evaluate_probs(eval_final_probs, target_data.y)
                if (
                    apd_eval_gate_sweep_gates
                    and m3_eval_active
                    and epoch % args.apd_eval_gate_sweep_interval == 0
                ):
                    needs_sweep_entropy = any(
                        state['score_mode'] in (
                            'idea_entropy', 'idea_joint', 'idea_calibrated'
                        )
                        for state in apd_eval_gate_sweep_state.values()
                    )
                    sweep_source_entropy = None
                    if needs_sweep_entropy:
                        sweep_source_entropy = apd_build_gate_source_entropy(
                            apd_cache['source_probs']
                        )
                    for sweep_name, sweep_gate_model in apd_eval_gate_sweep_gates.items():
                        sweep_state = apd_eval_gate_sweep_state[sweep_name]
                        sweep_entropy = (
                            sweep_source_entropy
                            if sweep_state['score_mode'] in (
                                'idea_entropy', 'idea_joint', 'idea_calibrated'
                            )
                            else None
                        )
                        sweep_gate = sweep_gate_model(
                            apd_cache['reliability'],
                            apd_cache['difficulty'],
                            source_entropy=sweep_entropy,
                            p_ata=p_ata,
                            source_mean_probs=eval_source_mean_probs,
                        )
                        sweep_final_probs, _ = apd_fuse_moe_predictions(
                            p_ata,
                            apd_cache['source_probs'],
                            sweep_gate,
                            source_temperature=args.apd_m3_source_temp,
                            ata_temperature=args.apd_m3_ata_temp,
                            debug=False,
                        )
                        apd_assert_probability(
                            'sweep_final_probs_{}'.format(sweep_name),
                            sweep_final_probs,
                            (target_data.num_nodes, args.num_classes),
                        )
                        sweep_acc, _, _, _ = apd_evaluate_probs(
                            sweep_final_probs,
                            target_data.y,
                        )
                        sweep_state['final_acc'] = float(sweep_acc)
                        if sweep_acc > sweep_state['best_acc']:
                            sweep_state['best_acc'] = float(sweep_acc)
                            sweep_state['best_epoch'] = epoch + 1
                if args.apd_source_gate_diag:
                    apd_last_source_gate_diag = apd_source_gate_diagnostics(
                        p_ata,
                        eval_source_mean_probs,
                        eval_final_probs,
                        eval_gate,
                        target_data.y,
                    )
        
        if args.apd_memory_update_mode == 'ata':
            
            outputs_target = p_ata.pow(2) / p_ata.pow(2).sum(dim=0, keepdim=True).clamp_min(1e-12)
            apd_assert_finite('outputs_target_ata', outputs_target)
        else:
            final_for_memory = None
            if args.apd_m3 and args.apd_eval_final and epoch >= args.apd_m3_start_epoch:
                final_for_memory = eval_final_probs
            memory_consensus = apd_select_reference_probs(
                p_ata,
                p_probe=apd_cache.get('p_probe'),
                p_final=final_for_memory,
                mode=args.apd_memory_update_mode,
                probe_weight=args.apd_memory_probe_weight,
            )
            outputs_target = apd_sharpen_memory_probs(memory_consensus)
        mem_cls = (1.0 - args.momentum) * mem_cls + args.momentum * outputs_target.clone()
        apd_assert_finite('mem_cls_after_update', mem_cls)
        if args.apd_memory_update_mode != 'ata':
            mem_cls = mem_cls / mem_cls.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * feat_output.clone()

        if args.apd_m3 and args.apd_eval_final:
            test_acc = test_acc_final
        else:
            test_acc, _, _, _ = evaluate(target_data, model)
        is_new_best = test_acc > best_acc
        if is_new_best:
            best_acc = test_acc
            best_epoch = epoch + 1
            if args.apd_best_artifact_path:
                best_artifact = {
                    'target': args.target,
                    'seed': args.seed,
                    'epoch': epoch + 1,
                    'acc': float(test_acc),
                    'source_names': list(src),
                    'labels': target_data.y.detach().cpu().clone(),
                    'p_ata': p_ata.detach().cpu().clone(),
                    'pseudo_y': pseudo_y.detach().cpu().clone(),
                    'pseudo_dist': pseudo_dist.detach().cpu().clone(),
                    'args': dict(vars(args)),
                }
                if args.apd_m1:
                    best_artifact.update({
                        'p_probe': apd_cache['p_probe'].detach().cpu().clone(),
                        'r_node': apd_cache['r_node'].detach().cpu().clone(),
                        'r_probe': apd_cache['r_probe'].detach().cpu().clone(),
                        'gap_score': apd_cache['gap_score'].detach().cpu().clone(),
                        'source_probs': apd_cache['source_probs'].detach().cpu().clone(),
                    })
                if args.apd_m2:
                    best_artifact.update({
                        'node_wrong_rate': node_wrong_rate.detach().cpu().clone(),
                        'expert_wrong_rate': expert_wrong_rate.detach().cpu().clone(),
                        'q_clean': q_clean.detach().cpu().clone(),
                        'q_noise': q_noise.detach().cpu().clone(),
                        'q_diff': q_diff.detach().cpu().clone(),
                        'reliability': apd_cache['reliability'].detach().cpu().clone(),
                        'difficulty': apd_cache['difficulty'].detach().cpu().clone(),
                    })
                if args.apd_m3 and args.apd_eval_final:
                    best_artifact.update({
                        'final_probs': eval_final_probs.detach().cpu().clone(),
                        'gate': eval_gate.detach().cpu().clone(),
                        'source_mean_probs': eval_source_mean_probs.detach().cpu().clone(),
                    })
        if args.apd_m3 and (args.apd_m3_eval_only or epoch >= args.apd_m3_start_epoch):
            if test_acc > best_m3_active_acc:
                best_m3_active_acc = test_acc
                best_m3_active_epoch = epoch + 1
        if args.apd_m3 and args.apd_eval_final:
            epoch_log_items = [
                'Epoch: {:04d}'.format(epoch + 1),
                'loss: {:.6f}'.format(loss),
                'accuracy: {:.6f}'.format(test_acc),
                'accuracy_p_ata: {:.6f}'.format(test_acc_p_ata),
                'accuracy_final: {:.6f}'.format(test_acc_final),
                'eval: final',
                'time: {:.6f}s'.format(time.time() - t),
            ]
            if args.apd_safe_blend:
                assert apd_last_safe_blend is not None
                epoch_log_items.extend([
                    'safe_allow_rate: {:.6f}'.format(apd_last_safe_blend['safe_allow_rate']),
                    'safe_pred_change_rate: {:.6f}'.format(apd_last_safe_blend['safe_pred_change_rate']),
                    'safe_final_acc: {:.6f}'.format(apd_last_safe_blend['safe_final_acc']),
                    'safe_final_minus_p_ata: {:.6f}'.format(apd_last_safe_blend['safe_final_minus_p_ata']),
                    'safe_fix_rate: {:.6f}'.format(apd_last_safe_blend['safe_fix_rate']),
                    'safe_break_rate: {:.6f}'.format(apd_last_safe_blend['safe_break_rate']),
                ])
            print(*epoch_log_items)
        else:
            print('Epoch: {:04d}'.format(epoch + 1), 'loss: {:.6f}'.format(loss), 'accuracy: {:.6f}'.format(test_acc), 'time: {:.6f}s'.format(time.time() - t))
    print('Optimization Finished!')
    if args.apd_best_artifact_path:
        assert best_artifact is not None, 'best artifact was requested but never captured'
        artifact_dir = os.path.dirname(os.path.abspath(args.apd_best_artifact_path))
        os.makedirs(artifact_dir, exist_ok=True)
        torch.save(best_artifact, args.apd_best_artifact_path)
        print(
            'APD_BEST_ARTIFACT path={} epoch={} acc={:.6f}'.format(
                args.apd_best_artifact_path,
                best_artifact['epoch'],
                best_artifact['acc'],
            )
        )
    print('Best observed target accuracy = {:.6f} at epoch = {:04d}'.format(best_acc, best_epoch))
    if args.apd_m3:
        print(
            'Best M3-active target accuracy = {:.6f} at epoch = {:04d}'.format(
                best_m3_active_acc,
                best_m3_active_epoch,
            )
        )
    for sweep_name, sweep_state in apd_eval_gate_sweep_state.items():
        print(
            'APD_RDEA_SWEEP_SUMMARY name={} score_mode={} difficulty_weight={:.6f} '
            'ata_floor={:.6f} confidence_weight={:.6f} best_acc={:.6f} '
            'best_epoch={} final_acc={:.6f}'.format(
                sweep_name,
                sweep_state['score_mode'],
                sweep_state['difficulty_weight'],
                sweep_state['ata_floor'],
                sweep_state['confidence_weight'],
                sweep_state['best_acc'],
                sweep_state['best_epoch'],
                sweep_state['final_acc'],
            )
        )
    if args.apd_m1:
        assert apd_last_r_node_mean is not None
        assert apd_last_loss_probe is not None
        print(
            'APD_M1_SUMMARY r_node_mean={:.6f} probe_loss={:.6f} probe_rng_mode={}'.format(
                apd_last_r_node_mean,
                apd_last_loss_probe,
                args.source_reliability_rng_mode,
            )
        )
    if args.apd_m2:
        assert apd_last_m2 is not None
        print(
            'APD_M2_SUMMARY node_wrong_rate_mean={:.6f} node_wrong_rate_std={:.6f} q_clean_mean={:.6f} q_noise_mean={:.6f} q_diff_mean={:.6f} difficulty_mean={:.6f} difficulty_std={:.6f} loss_clean={:.6f} lambda_clean={:.6f} loss_noise={:.6f} loss_sim={:.6f} sim_view_mode={} sim_view_count={} weight_mode={} wrong_signal={} wrong_momentum={:.6f} wrong_event_mode={} expert_wrong_signal={} expert_wrong_momentum={:.6f} wrong_norm={} reliability_mode={} gap_floor_mode={} gap_floor_mix={:.6f} m1_m2_coupling={} coupling_active={} soft_target_mode={} source_prune_mode={} source_prune_floor={:.6f} pwed_prior_mix={:.6f} marginal_prior_l1_uniform={:.6f}'.format(
                apd_last_m2['node_wrong_rate_mean'],
                apd_last_m2['node_wrong_rate_std'],
                apd_last_m2['q_clean_mean'],
                apd_last_m2['q_noise_mean'],
                apd_last_m2['q_diff_mean'],
                apd_last_m2['difficulty_mean'],
                apd_last_m2['difficulty_std'],
                apd_last_m2['loss_clean'],
                apd_last_m2['lambda_clean'],
                apd_last_m2['loss_noise'],
                apd_last_m2['loss_sim'],
                apd_last_m2['sim_view_mode'],
                apd_last_m2['sim_view_count'],
                apd_last_m2['weight_mode'],
                args.apd_wrong_signal,
                args.apd_wrong_momentum,
                args.apd_wrong_event_mode,
                args.apd_expert_wrong_signal,
                args.apd_expert_wrong_momentum,
                args.apd_wrong_norm,
                args.apd_reliability_mode,
                args.apd_gap_floor_mode,
                args.apd_gap_floor_mix,
                apd_last_m2['m1_m2_coupling'],
                apd_last_m2['coupling_active'],
                apd_last_m2['soft_target_mode'],
                apd_last_m2['source_prune_mode'],
                apd_last_m2['source_prune_floor'],
                apd_last_m2['pwed_prior_mix'],
                apd_last_m2['marginal_prior_l1_uniform'],
            )
        )
    if args.apd_m3:
        assert apd_last_m3 is not None
        print(
            'APD_M3_SUMMARY m3_mode={} gate_type={} gate_score_mode={} gate_source_sharpness={:.6f} gate_difficulty_weight={:.6f} gate_confidence_weight={:.6f} gate_ata_mean={:.6f} gate_ata_min={:.6f} gate_ata_max={:.6f} gate_source_mean_list={} loss_clean={:.6f} loss_noise={:.6f} loss_reg={:.6f} loss_anchor={:.6f} loss_gate_prior={:.6f} loss_total={:.6f} loss_m2_base={:.6f} loss_train={:.6f} train_mode={} m3_train_active={} m3_start_epoch={} lambda_m3={:.6f} source_temp={:.6f} ata_temp={:.6f} clean_weight_mode={} memory_update_mode={} eval_blend_mode={} eval_blend_gamma={:.6f} safe_blend={} safe_source_mix={} safe_blend_gamma={:.6f} safe_conf_margin={:.6f} safe_pseudo_margin={:.6f}'.format(
                apd_last_m3['m3_mode'],
                apd_last_m3['gate_type'],
                apd_last_m3['gate_score_mode'],
                apd_last_m3['gate_source_sharpness'],
                apd_last_m3['gate_difficulty_weight'],
                apd_last_m3['gate_confidence_weight'],
                apd_last_m3['gate_ata_mean'],
                apd_last_m3['gate_ata_min'],
                apd_last_m3['gate_ata_max'],
                ','.join('{:.6f}'.format(value) for value in apd_last_m3['gate_source_mean']),
                apd_last_m3['loss_clean'],
                apd_last_m3['loss_noise'],
                apd_last_m3['loss_reg'],
                apd_last_m3['loss_anchor'],
                apd_last_m3['loss_gate_prior'],
                apd_last_m3['loss_total'],
                apd_last_m3['loss_m2_base'],
                apd_last_m3['loss_train'],
                apd_last_m3['train_mode'],
                apd_last_m3['m3_train_active'],
                apd_last_m3['m3_start_epoch'],
                apd_last_m3['lambda_m3'],
                apd_last_m3['source_temp'],
                apd_last_m3['ata_temp'],
                apd_last_m3['clean_weight_mode'],
                args.apd_memory_update_mode,
                apd_last_m3['eval_blend_mode'],
                apd_last_m3['eval_blend_gamma'],
                apd_last_m3['safe_blend'],
                apd_last_m3['safe_source_mix'],
                apd_last_m3['safe_blend_gamma'],
                apd_last_m3['safe_conf_margin'],
                apd_last_m3['safe_pseudo_margin'],
            )
        )
    if args.apd_safe_blend:
        assert apd_last_safe_blend is not None
        print(
            'APD_SAFE_BLEND_SUMMARY safe_source_mix={} safe_blend_gamma={:.6f} safe_conf_margin={:.6f} safe_pseudo_margin={:.6f} safe_allow_rate={:.6f} safe_pred_change_rate={:.6f} safe_final_acc={:.6f} safe_final_minus_p_ata={:.6f} safe_fix_rate={:.6f} safe_break_rate={:.6f}'.format(
                args.apd_safe_source_mix,
                args.apd_safe_blend_gamma,
                args.apd_safe_conf_margin,
                args.apd_safe_pseudo_margin,
                apd_last_safe_blend['safe_allow_rate'],
                apd_last_safe_blend['safe_pred_change_rate'],
                apd_last_safe_blend['safe_final_acc'],
                apd_last_safe_blend['safe_final_minus_p_ata'],
                apd_last_safe_blend['safe_fix_rate'],
                apd_last_safe_blend['safe_break_rate'],
            )
        )
    if args.apd_source_gate_diag:
        assert apd_last_source_gate_diag is not None
        print(
            'APD_SOURCE_GATE_DIAG source_names={} source_acc_list={} source_best_idx={} source_best_acc={:.6f} source_best_fixable_rate={:.6f} source_best_break_rate={:.6f} source_best_change_rate={:.6f} source_mean_acc={:.6f} source_mean_change_rate={:.6f} source_gate_mix_acc={:.6f} source_gate_mix_change_rate={:.6f} source_oracle_acc={:.6f} source_oracle_fixable_rate={:.6f} expert_oracle_acc={:.6f} expert_oracle_gain={:.6f} selected_source_acc={:.6f} selected_source_fixable_rate={:.6f} selected_source_break_rate={:.6f} selected_source_change_rate={:.6f} gate_ata_mean={:.6f} gate_source_mass_mean={:.6f} gate_non_ata_argmax_rate={:.6f} pred_change_rate={:.6f} final_fix_rate={:.6f} final_break_rate={:.6f} p_ata_acc={:.6f} final_acc={:.6f} final_minus_p_ata={:.6f} blend_source_mean_metrics={} blend_source_gate_metrics={} blend_source_best_metrics={}'.format(
                ','.join(src),
                ','.join('{:.6f}'.format(value) for value in apd_last_source_gate_diag['source_acc_list']),
                apd_last_source_gate_diag['source_best_idx'],
                apd_last_source_gate_diag['source_best_acc'],
                apd_last_source_gate_diag['source_best_fixable_rate'],
                apd_last_source_gate_diag['source_best_break_rate'],
                apd_last_source_gate_diag['source_best_change_rate'],
                apd_last_source_gate_diag['source_mean_acc'],
                apd_last_source_gate_diag['source_mean_change_rate'],
                apd_last_source_gate_diag['source_gate_mix_acc'],
                apd_last_source_gate_diag['source_gate_mix_change_rate'],
                apd_last_source_gate_diag['source_oracle_acc'],
                apd_last_source_gate_diag['source_oracle_fixable_rate'],
                apd_last_source_gate_diag['expert_oracle_acc'],
                apd_last_source_gate_diag['expert_oracle_gain'],
                apd_last_source_gate_diag['selected_source_acc'],
                apd_last_source_gate_diag['selected_source_fixable_rate'],
                apd_last_source_gate_diag['selected_source_break_rate'],
                apd_last_source_gate_diag['selected_source_change_rate'],
                apd_last_source_gate_diag['gate_ata_mean'],
                apd_last_source_gate_diag['gate_source_mass_mean'],
                apd_last_source_gate_diag['gate_non_ata_argmax_rate'],
                apd_last_source_gate_diag['pred_change_rate'],
                apd_last_source_gate_diag['final_fix_rate'],
                apd_last_source_gate_diag['final_break_rate'],
                apd_last_source_gate_diag['p_ata_acc'],
                apd_last_source_gate_diag['final_acc'],
                apd_last_source_gate_diag['final_minus_p_ata'],
                apd_last_source_gate_diag['blend_source_mean_metrics'],
                apd_last_source_gate_diag['blend_source_gate_metrics'],
                apd_last_source_gate_diag['blend_source_best_metrics'],
            )
        )


    return test_acc, best_acc, best_epoch


if __name__ == '__main__':
    test_acc, best_acc, best_epoch = train_target(data)
    print('After adaptation, test set results accuracy = {:.6f}'.format(test_acc))
    print('Best observed target accuracy = {:.6f} at epoch = {:04d}'.format(best_acc, best_epoch))

    results_parent = os.path.dirname(os.path.abspath(args.results_path))
    os.makedirs(results_parent, exist_ok=True)
    with open(args.results_path, 'a+') as f:
        f.write(args.target + ',' + str(test_acc) + ',' + str(best_acc) + ',' + str(best_epoch) + '\n')

    print('*'*50)
    print('Before adaptation, the performance of each model is as follows: ')
    for i in range(len(src)):
        model = NodeClassificationModel(args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, 'model_' + src[i] + '.pth')))
        test_acc, macro_f1, micro_f1, pretrain_test_loss = evaluate(data, model)
        print('Source: ' + src[i] + ', ' + 'accuracy: ' + str(test_acc))
