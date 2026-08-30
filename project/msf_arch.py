"""Derive a MedSymmFlow UNet's architecture from its checkpoint state dict.

The published checkpoints were trained with non-default architecture flags, and
passing the argparser defaults silently produces a shape mismatch on load (or,
worse, a size mismatch only at the skip connections). Reading the shapes is the
only reliable source, so every script here infers the architecture instead of
hard-coding it.

Derivation (verified against the RGB_28 pneumoniamnist checkpoint):
  model_channels  = input_blocks.0.0.weight.shape[0]
  in_channels     = input_blocks.0.0.weight.shape[1]   (1 image + 3 RGB mask = 4)
  res blocks      = input_blocks.<i>.0.out_layers.3.weight, grouped by index gaps
                    (downsample blocks sit in the gaps)
  channel_mult    = first width of each group / model_channels
  attention       = groups containing input_blocks.<i>.1.qkv -> ds = 2**level

num_head_channels does NOT affect any tensor shape (qkv is always 3*ch), so it
cannot be inferred; it defaults to the value used in the repo's own commands.
"""
import re

import torch

DEFAULT_NUM_HEAD_CHANNELS = 64
DEFAULT_NUM_HEADS = 4


def load_state_dict(checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("model_state", "state_dict", "ema", "model"):
            inner = sd.get(key)
            if isinstance(inner, dict) and any(hasattr(v, "shape") for v in inner.values()):
                return inner
    return sd


def infer_arch(checkpoint_path_or_sd, num_head_channels=DEFAULT_NUM_HEAD_CHANNELS,
               num_heads=DEFAULT_NUM_HEADS):
    sd = (checkpoint_path_or_sd if isinstance(checkpoint_path_or_sd, dict)
          else load_state_dict(checkpoint_path_or_sd))

    stem = sd["input_blocks.0.0.weight"]
    model_channels, in_channels = int(stem.shape[0]), int(stem.shape[1])

    widths = {}
    for k, v in sd.items():
        m = re.fullmatch(r"input_blocks\.(\d+)\.0\.out_layers\.3\.weight", k)
        if m:
            widths[int(m.group(1))] = int(v.shape[0])
    if not widths:
        raise ValueError("No residual blocks found; is this a MedSymmFlow UNet checkpoint?")

    # Consecutive indices belong to one resolution level; gaps are downsample blocks.
    idx = sorted(widths)
    groups, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)

    num_res_blocks = len(groups[0])
    channel_mult = [widths[g[0]] // model_channels for g in groups]

    attn_blocks = {int(re.findall(r"\d+", k)[0])
                   for k in sd if re.match(r"input_blocks\.\d+\.1\.qkv", k)}
    attention_resolutions = sorted({2 ** lvl for lvl, g in enumerate(groups)
                                    if attn_blocks & set(g)}) or [2]

    return {
        "model_channels": model_channels,
        "in_channels": in_channels,
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "attention_resolutions": attention_resolutions,
        "num_heads": num_heads,
        "num_head_channels": num_head_channels,
        "rgb_mask": in_channels >= 4,
    }


def apply_arch(model_args, arch):
    """Copy an inferred architecture onto a parsed MedSymmFlow args namespace."""
    model_args.model_channels = arch["model_channels"]
    model_args.num_res_blocks = arch["num_res_blocks"]
    model_args.channel_mult = tuple(arch["channel_mult"])
    model_args.attention_resolutions = tuple(arch["attention_resolutions"])
    model_args.num_heads = arch["num_heads"]
    model_args.num_head_channels = arch["num_head_channels"]
    return model_args


def add_arch_cli(parser):
    parser.add_argument("--auto_arch", action="store_true", default=True,
                        help="Infer UNet architecture from the checkpoint (default)")
    parser.add_argument("--no_auto_arch", dest="auto_arch", action="store_false")
    parser.add_argument("--model_channels", type=int, default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=None)
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=None)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num_head_channels", type=int, default=DEFAULT_NUM_HEAD_CHANNELS)
    return parser


def resolve_arch(args):
    """CLI overrides win; anything unset is inferred from the checkpoint."""
    arch = (infer_arch(args.checkpoint, args.num_head_channels, args.num_heads)
            if args.auto_arch else
            {"model_channels": 64, "in_channels": 4, "num_res_blocks": 2,
             "channel_mult": [1, 2, 2, 2], "attention_resolutions": [2],
             "num_heads": args.num_heads, "num_head_channels": args.num_head_channels,
             "rgb_mask": True})
    for key in ("model_channels", "num_res_blocks", "channel_mult", "attention_resolutions"):
        if getattr(args, key, None) is not None:
            arch[key] = getattr(args, key)
    return arch
