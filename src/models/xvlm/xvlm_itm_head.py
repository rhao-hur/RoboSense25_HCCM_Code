# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.evaluation import Accuracy
from mmpretrain.registry import MODELS


class Pooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@MODELS.register_module()
class XVLM_ITMHead(BaseModule):
    def __init__(self,
                 hidden_size: int,
                 with_pooler: bool = True,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(XVLM_ITMHead, self).__init__(init_cfg=init_cfg)
        self.hidden_size = hidden_size

        if with_pooler:
            self.pooler = Pooler(hidden_size=self.hidden_size)
        else:
            self.pooler = nn.Identity()
            
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, 2)
        )

        self.loss_module = MODELS.build(loss)
        self.cal_acc = cal_acc

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pooler(feats[-1])
        itm_logits = self.fc(pre_logits)
        return itm_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples, **kwargs) -> dict:

        itm_logits = self(feats)

        if itm_logits.ndim == 3:
            itm_logits = itm_logits.mean(dim=1)

        losses = self._get_loss(itm_logits, data_samples, **kwargs)
        return losses

    def _get_loss(self, itm_logits: torch.Tensor, data_samples, **kwargs):
        target = torch.tensor([i.is_matched
                               for i in data_samples]).to(itm_logits.device)

        losses = dict()

        loss = self.loss_module(
            itm_logits, target.long(), avg_factor=itm_logits.size(0), **kwargs)
        losses['itm_loss'] = loss

        if self.cal_acc:
            acc = Accuracy.calculate(itm_logits, target)
            losses.update({'itm_accuracy': acc[0][0]})

        return losses
