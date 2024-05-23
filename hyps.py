import dataclasses
from typing import Literal


@dataclasses.dataclass
class Hyps:
    stem_ksize: int = 5
    stem_feats: int = 32
    stem_type: Literal["max_pool", "conv"] = "conv"

    bottleneck: float = 1
    main_ksize: int = 3
    feats_gain: float = 2
    lvl_blocks: int = 2
    lvl_count: int = 2

    head_type: Literal["avg_pool", "dense"] = "dense"
    head_hid_dim: int = 256

    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-2


MIN_HYPS = Hyps(
    stem_ksize=3,
    stem_feats=4,
    bottleneck=0.1,
    main_ksize=3,
    feats_gain=1.0,
    lvl_blocks=2,
    lvl_count=1,
    head_hid_dim=100,
    dropout=0.0,
    lr=1e-4,
    weight_decay=1e-2
)

MAX_HYPS = Hyps(
    stem_ksize=15,
    stem_feats=64,
    bottleneck=1,
    main_ksize=7,
    feats_gain=2.0,
    lvl_blocks=4,
    lvl_count=3,
    head_hid_dim=512,
    dropout=0.5,
    lr=1e-2,
    weight_decay=0.5
)
