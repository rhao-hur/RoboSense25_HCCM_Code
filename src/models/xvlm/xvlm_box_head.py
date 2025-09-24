# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.evaluation import Accuracy
from mmpretrain.registry import MODELS

@MODELS.register_module()
class XVLM_BOXHead(BaseModule):
    def __init__(self,
                 hidden_size: int,
                 init_cfg: Optional[dict] = None):
        super(XVLM_BOXHead, self).__init__(init_cfg=init_cfg)
        self.hidden_size = hidden_size
            
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, 4)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """The forward process."""
        return self.fc(feats)
