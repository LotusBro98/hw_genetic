from typing import Literal


class Hyps:
    stem_ksize: int = 7
    stem_feats: int = 64
    stem_type: Literal["max_pool", "conv"] = "conv"

    bottleneck: float = 1
    main_ksize: int = 3
    feats_gain: float = 2
    lvl_blocks = [2, 2, 2, 2, 2]
    lvl_count: int = 2

    head_type: Literal["avg_pool", "dense"] = "dense"
    head_hid_dim: int = 512

    dropout: float = 0.0
    lr: float = 1e-3
    lr_scheduler: Literal["linear", "exp", "step"]
    lr_sched_period: int