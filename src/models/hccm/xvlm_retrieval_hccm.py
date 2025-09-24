from collections import ChainMap
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
import gc
import time

import mmengine.dist as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import distributed as torch_dist
from torch.cuda.amp import autocast
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from mmpretrain.utils import track_on_main_process
from ..xvlm.xvlm_xbert_tokenizer import BertTokenizer
from ..utils import box_ops
import re
from torchvision.ops import roi_align

def all_gather_concat(data: torch.Tensor) -> torch.Tensor:

    if dist.get_world_size() == 1:
        return data

    data_size = torch.tensor(data.size(0), device=data.device)
    sizes_list = dist.all_gather(data_size)

    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = dist.all_gather(data)

    all_data = []
    for tensor, size in zip(gather_list, sizes_list):

        all_data.append(tensor[:size])

    return torch.concat(all_data)


@MODELS.register_module()
class Linear(torch.nn.Linear):
     """Wrapper for linear function."""


@MODELS.register_module()
class XVLMRetrieval_hccm(BaseModel):

    def __init__(self,
                 vision_encoder: dict,
                 text_encoder: dict,
                 vision_proj: Optional[dict] = None,
                 text_proj: Optional[dict] = None,
                 itm_head: Optional[Union[List[dict], dict]] = None,
                 itc_head: Optional[Union[List[dict], dict]] = None,
                 bbox_head: Optional[Union[List[dict], dict]] = None,
                 tokenizer_path: str = None,
                 momentum: float = .995,
                 negative_all_rank: bool = True,
                 temperature: float = 0.07,
                 fast_match: bool = False,
                 topk: int = 256,
                 entis_itc_distill: bool = False,
                 alpha: float = 0.4,
                 max_tokens: int = 20,
                 train_max_words: int = 20,
                 val_max_words: int = 20,
                 w_itc: float = 0.25,
                 w_itc_entis: float = 0.25,
                 w_itm: float = 1,
                 w_itm_entis: float = 0.5,
                 w_box: float = 0.1,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.vision_encoder = MODELS.build(vision_encoder)
        self.text_encoder = MODELS.build(text_encoder)

        if vision_proj is not None:
            self.vision_proj = MODELS.build(vision_proj)

        if text_proj is not None:
            self.text_proj = MODELS.build(text_proj)

        if itm_head is not None:
            self.itm_head = MODELS.build(itm_head)

        if itc_head is not None:
            self.itc_head = MODELS.build(itc_head)

        if bbox_head is not None:
            self.bbox_head = MODELS.build(bbox_head)

        if tokenizer_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.momentum = momentum
        self.negative_all_rank = negative_all_rank
        self.temp = nn.Parameter(temperature * torch.ones([]))

        # create the momentum encoder
        self.vision_encoder_m = deepcopy(self.vision_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.vision_encoder, self.vision_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        self.fast_match = fast_match
        self.topk = topk

        self.entis_itc_distill = entis_itc_distill
        self.alpha = alpha

        self.train_max_words = train_max_words
        self.val_max_words = val_max_words
        self.max_tokens = max_tokens

        self.w_itc = w_itc
        self.w_itm = w_itm
        self.w_itc_entis = w_itc_entis
        self.w_itm_entis = w_itm_entis
        self.w_box = w_box

    @property
    def device(self):
        return next(self.parameters()).device
    
    def pre_caption(self, caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption
    
    def xywh2xyxy(self, x):
        assert isinstance(x, torch.Tensor), "x should be a torch tensor."
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    
    def preprocess_datasamples(self, data_samples):
        if self.training:
            tokenizer_padding = 'longest'
            truncation = False
            max_words = self.train_max_words
            texts_words_len = [len(sample.get('text').split(' ')) for sample in data_samples]
        else:
            tokenizer_padding = 'max_length'
            truncation = True
            max_words = self.val_max_words
            
        pair_sens_bbox = {}
        texts = []
        for b, sample in enumerate(data_samples):
            if sample.get("bboxes") is not None and sample.get("sentences") is not None:
                pair_sens_bbox[b] = []

                for i in range(3): 
                    if sample.get("bboxes")[i] is not None:
                        target_bbox = torch.tensor(sample.get("bboxes")[i], dtype=torch.float32).to(self.device)
                        sen_token = self.tokenizer(
                            self.pre_caption(sample.get("sentences")[i], max_words),
                            padding=tokenizer_padding,
                            truncation=truncation,
                            max_length=self.max_tokens,
                            return_tensors='pt'
                        ).to(self.device)
                        pair_sens_bbox[b].append({'bbox': target_bbox, 'sen_token': sen_token})
                
                # =========== auto words ===========
                if None not in sample.get("sentences"):
                    sens_len = 0
                    for sens in sample.get("sentences"):
                        sens_len += len(sens.split(" "))
                    texts_words_len[b] -= sens_len
                else:
                    texts_words_len[b] = 50

                    
            if sample.get("text") is not None:
                if isinstance(sample.get('text'), (list, tuple)):
                    texts.extend(sample.get('text'))
                elif isinstance(sample.get('text'), str):
                    texts.append(sample.get('text'))
                else:
                    raise TypeError('text must be a string or a list of strings')

        if self.training:
            texts = [self.pre_caption(t, texts_words_len[j]) for j, t in enumerate(texts)]
        texts = self.tokenizer(
            [self.pre_caption(t, max_words) for t in texts],
            padding=tokenizer_padding,
            truncation=truncation,
            max_length=self.max_tokens,
            return_tensors='pt',
        ).to(self.device)


        # ================= entis_pair process =================
        texts_tokens_len = texts.input_ids.size(1)
        entis_pair = []
        for bid, sample in enumerate(data_samples):
            if sample.get("bboxes") is not None and sample.get("sentences") is not None:
                bboxes = sample.get("bboxes")
                sentences = sample.get("sentences")

                valid_bboxes = []
                valid_sens = []

                for i in range(3):
                    if bboxes[i] is not None and sentences[i] is not None:
                        valid_bboxes.append([bid] + bboxes[i])
                        valid_sens.append(sentences[i])
                
                if valid_bboxes and valid_sens:
                    boxes_tensor = self.xywh2xyxy(torch.Tensor(valid_bboxes)).to(self.device)
                    entis_tokens = self.tokenizer(
                        [self.pre_caption(s, max_words) for s in valid_sens],
                        padding='max_length',
                        truncation=True,
                        max_length=texts_tokens_len,
                        return_tensors='pt'
                    ).to(self.device)
                else:
                    boxes_tensor = None
                    entis_tokens = None
                
                entis_pair.append({'boxes': boxes_tensor, 'entis_text': entis_tokens})

        return texts, pair_sens_bbox, entis_pair

    def forward(self,
                images: torch.tensor = None,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor') -> Union[Tuple, dict]:
        """The unified entry for a forward process in both training and test.
        The method should accept two modes: "tensor", and "loss":

        - "tensor": Forward the whole network and return tensor without any
          post-processing, same as a common nn.Module.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        For unified "predict" mode in other mm repos. It is noticed that
        image-text retrieval cannot perform batch prediction since it will go
        through all the samples. A standard process of retrieval evaluation is
        to extract and collect all feats, and then predict all samples.
        Therefore the `predict` mode here is remained as a trigger
        to inform use to choose the right configurations.

        Args:
            images (torch.Tensor): The input inputs tensor of shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="tensor"``, return a tuple.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            return self.extract_feat(images, data_samples)
        elif mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
    
    def extract_feat(
        self,
        images: torch.Tensor = None,
        data_samples: List[DataSample] = None,
        return_texts=True,
        return_embeds=None,
    ) -> Dict[str, torch.Tensor]:

        valid_samples = [ds for ds in data_samples if ds.all_keys()]
        if valid_samples:
            texts, pair_sens_bbox, entis_pair = self.preprocess_datasamples(data_samples)
        else:
            texts = None
            pair_sens_bbox = None
            entis_pair = None

        assert images is not None or texts is not None, \
            'At least single modality should be passed as inputs.'

        results = {}
        if texts is not None and return_texts:
            results.update({
                'text_ids': texts.input_ids,
                'text_attn_mask': texts.attention_mask,
            })

        if pair_sens_bbox is not None and self.training:
            results.update({
                'pair_sens_bbox': pair_sens_bbox,
            })
        
        if entis_pair is not None and self.training:
            results.update({
                'entis_pair': entis_pair,
            })

        if return_embeds is None:
            return_embeds = not self.fast_match

        # extract image features
        if images is not None:
            output = self._extract_feat(images, modality='images')
            results['image_feat'] = output['image_feat']
            if return_embeds:
                results['image_embeds'] = output['image_embeds']

        # extract text features
        if texts is not None:
            output = self._extract_feat(texts, modality='texts')
            results['text_feat'] = output['text_feat']
            if return_embeds:
                results['text_embeds'] = output['text_embeds']

        return results

    def _extract_feat(self, inputs: Union[torch.Tensor, dict],
                      modality: str) -> Tuple[torch.Tensor]:


        if modality == 'images':
            # extract image features
            image_embeds = self.vision_encoder(inputs)
            image_feat = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            return {'image_embeds': image_embeds, 'image_feat': image_feat}
        elif modality == 'texts':
            # extract text features
            text_output = self.text_encoder(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_type_ids=None,
                return_dict=True,
                mode='text',
            )
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(
                self.text_proj(text_embeds[:, 0, :]), dim=-1)
            return {'text_embeds': text_embeds, 'text_feat': text_feat}
        else:
            raise RuntimeError(f'Invalid modality "{modality}".')
        
    def compute_bbox_loss(self, output_coord, target_bbox, is_image=None):
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4
        
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
    
    def get_entis_contrastive_loss(
            self, all_entis_vision_feat, all_entis_text_feat,
            valid_image_feat_m, valid_text_feat_m,
            all_entis_idx, valid_idx):
        
        valid_idx = valid_idx.view(-1, 1).to(self.device)
        pos_idx = torch.eq(all_entis_idx.view(-1, 1), valid_idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_i2t = all_entis_vision_feat @ valid_text_feat_m.t() / self.temp
        sim_t2i = all_entis_text_feat @ valid_image_feat_m.t() / self.temp


        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()


        return {"entis_itc_loss":  (loss_i2t + loss_t2i) * self.w_itc_entis}
    
    def loss(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
    ) -> Dict[str, torch.tensor]:

        output = self.extract_feat(images, data_samples, return_embeds=True)

        text_ids = output['text_ids']
        text_attn_mask = output['text_attn_mask']
        image_embeds = output['image_embeds']
        image_feat = output['image_feat']
        text_feat = output['text_feat']
        text_embeds = output["text_embeds"]
        pair = output['pair_sens_bbox']
        entis_pair = output['entis_pair']
        

        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
        loss = {}

        
        # ============= compute the contrast loss ==============
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.vision_encoder_m(images)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

            text_embeds_m_raw = self.text_encoder_m(
                text_ids,
                attention_mask=text_attn_mask,
                token_type_ids=None,
                return_dict=True,
                mode='text',
            ).last_hidden_state
            text_feat_m = F.normalize(
                self.text_proj_m(text_embeds_m_raw[:, 0, :]), dim=-1)

        itc_loss = self.itc_head.loss(
            ([image_feat, text_feat, image_feat_m, text_feat_m], ),
            data_samples, self.temp)
        itc_loss['itc_loss'] = itc_loss['itc_loss'] * self.w_itc
        loss.update(itc_loss)
        
        # ============== compute the entity loss ===============
        all_entis_text_embeds, all_entis_text_attention_mask = [], []
        all_entis_vision_embeds, all_entis_vision_attention_mask = [], []
        all_entis_text_feat, all_entis_vision_feat = [], []
        all_entis_idx, repeat_counts = [], []
        valid_indices = []
        
        for bs_id, pair_info in enumerate(entis_pair):
            if pair_info['boxes'] is None: continue

            valid_indices.append(bs_id)

            # -------------- create entis_vision_feat --------------
            entis = roi_align(
                input = images, 
                boxes = pair_info['boxes'], 
                output_size = images.size(-1)
            )
            entis_embeds = self.vision_encoder(entis)
            all_entis_vision_embeds.append(entis_embeds)
            all_entis_vision_attention_mask.append(
                torch.ones(entis_embeds.size()[:-1], dtype=torch.long).to(self.device))
            entis_vision_feat = F.normalize(self.vision_proj(entis_embeds[:, 0, :]))


            # -------------- create entis_text_feat --------------
            entis_input_ids = pair_info['entis_text'].input_ids
            entis_attention_mask = pair_info['entis_text'].attention_mask
            entis_text_embeds = self.text_encoder(
                entis_input_ids,
                attention_mask=entis_attention_mask,
                token_type_ids=None,
                return_dict=True,
                mode='text',
            ).last_hidden_state
            all_entis_text_embeds.append(entis_text_embeds)
            all_entis_text_attention_mask.append(pair_info['entis_text'].attention_mask)
            entis_text_feat = F.normalize(self.text_proj(entis_text_embeds[:, 0, :]))

            num_pairs = entis.size(0)
            repeat_counts.append(num_pairs)

            all_entis_text_feat.append(entis_text_feat)
            all_entis_vision_feat.append(entis_vision_feat)
            all_entis_idx.extend([data_samples[bs_id].image_id] * entis.size(0))

        if all_entis_text_feat and all_entis_vision_feat:
            all_entis_text_feat = torch.concat(all_entis_text_feat, dim = 0)
            all_entis_vision_feat = torch.concat(all_entis_vision_feat, dim = 0)

            all_entis_idx = torch.tensor(all_entis_idx, device=images.device)
            
            valid_text_feat_m = text_feat_m[valid_indices]
            valid_image_feat_m = image_feat_m[valid_indices]
            valid_idx = torch.tensor([data_samples[i].image_id for i in valid_indices], device = images.device)

            repeat_counts = torch.tensor(repeat_counts, device=images.device)
            entis_bsidx = torch.tensor(
                valid_indices, device = self.device).repeat_interleave(repeat_counts, dim=0)
            all_entis_text_embeds = torch.concat(all_entis_text_embeds, dim = 0)
            all_entis_text_attention_mask = torch.concat(all_entis_text_attention_mask, dim = 0)
            all_entis_vision_embeds = torch.concat(all_entis_vision_embeds, dim = 0)
            all_entis_vision_attention_mask = torch.concat(all_entis_vision_attention_mask, dim = 0)
            
            entis_itc_loss = self.get_entis_contrastive_loss(
                all_entis_vision_feat, all_entis_text_feat,
                valid_image_feat_m, valid_text_feat_m,
                all_entis_idx, valid_idx
            )
            loss.update(entis_itc_loss)
        
        # =============== compute the bboxes ===============
        total_bb_loss = []
        for bs_id, bbox_info_list in pair.items():
            for bbox_info in bbox_info_list:
                sen_token = bbox_info['sen_token']
                target_bbox = bbox_info['bbox']
                sen_output = self.text_encoder(
                    sen_token.input_ids,
                    attention_mask=sen_token.attention_mask,
                    token_type_ids=None,
                    return_dict=True,
                    mode='text',
                )
                sen_embeds = sen_output.last_hidden_state

                output_cls = self.text_encoder(
                        encoder_embeds=sen_embeds,
                        attention_mask=sen_token.attention_mask,
                        encoder_hidden_states=image_embeds[bs_id].unsqueeze(0),
                        encoder_attention_mask=image_atts[bs_id].unsqueeze(0),
                        return_dict=True,
                        mode='fusion'
                    ).last_hidden_state[:, 0, :]
                output_coord = self.bbox_head(output_cls).sigmoid()
                loss_bbox, loss_giou = self.compute_bbox_loss(output_coord, target_bbox.unsqueeze(0))
                total_bb_loss.append(loss_bbox + loss_giou)

        total_bb_loss = self.w_box * torch.stack(total_bb_loss).mean() if total_bb_loss else None

        # =========== Prepare negative samples for matching loss compute ===========
        output_pos = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_attn_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion'
        )
        idx = torch.tensor([i.image_id for i in data_samples], device=self.device).view(-1, 1)
        bs = idx.size(0)
        idxs = torch.cat(dist.all_gather(idx))
        if self.negative_all_rank:
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t()).to(self.device)
                image_feat_world = torch.cat(dist.all_gather(image_feat))
                text_feat_world = torch.cat(dist.all_gather(text_feat))

                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            world_size = dist.get_world_size()
            if world_size == 1:
                image_embeds_world = image_embeds
            else:
                image_embeds_world = torch.cat(
                    torch_dist.nn.all_gather(image_embeds))
                
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            text_embeds_world = torch.cat(dist.all_gather(text_embeds))
            att_mask_world = torch.cat(dist.all_gather(text_attn_mask))

            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(text_embeds_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_attn_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode='fusion'
        )
        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        data_samples.extend(
            [DataSample(is_matched=False) for _ in range(2 * bs)])
        
        loss_multimodal = self.itm_head.loss((vl_embeddings, ), data_samples)
        loss_multimodal['itm_loss'] = loss_multimodal['itm_loss'] * self.w_itm
        

        # ============================ image and text entis matching ============================
        if not isinstance(all_entis_text_feat, List) and not isinstance(all_entis_vision_feat, List):
            entis_idx = all_entis_idx.view(-1, 1)
            bs = entis_idx.size(0)

            with torch.no_grad():            
                mask = torch.eq(entis_idx, idxs.t())

                sim_is2t = all_entis_vision_feat @ text_feat_world.t() / self.temp
                sim_ts2i = all_entis_text_feat @ image_feat_world.t() / self.temp
                
                weights_is2t = F.softmax(sim_is2t, dim=1)
                weights_is2t.masked_fill_(mask, 0)

                weights_ts2i = F.softmax(sim_ts2i, dim=1)
                weights_ts2i.masked_fill_(mask, 0)

            image_embeds_world = torch.cat(dist.all_gather(image_embeds))
            image_att_mask_world = torch.cat(dist.all_gather(image_atts))
            image_embeds_neg = []
            image_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_ts2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
                image_atts_neg.append(image_att_mask_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
            image_atts_neg = torch.stack(image_atts_neg, dim=0)

            text_embeds_world = torch.cat(dist.all_gather(text_embeds))
            text_att_mask_world = torch.cat(dist.all_gather(text_attn_mask))
            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_is2t[b], 1).item()
                text_embeds_neg.append(text_embeds_world[neg_idx])
                text_atts_neg.append(text_att_mask_world[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_embeds_all = torch.cat([all_entis_text_embeds, text_embeds_neg], dim=0)
            text_atts_all = torch.cat([all_entis_text_attention_mask, text_atts_neg], dim=0)

            image_embeds_all = torch.cat([image_embeds_neg, all_entis_vision_embeds], dim=0)
            image_atts_all = torch.cat([image_atts_neg, all_entis_vision_attention_mask], dim=0)

            output_neg_entis = self.text_encoder(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                mode='fusion'
            )

            output_pos_is = self.text_encoder(
                encoder_embeds=text_embeds[entis_bsidx],
                attention_mask=text_attn_mask[entis_bsidx],
                encoder_hidden_states=all_entis_vision_embeds,
                encoder_attention_mask=all_entis_vision_attention_mask,
                return_dict=True,
                mode='fusion'
            )
            output_pos_ts = self.text_encoder(
                encoder_embeds=all_entis_text_embeds,
                attention_mask=all_entis_text_attention_mask,
                encoder_hidden_states=image_embeds[entis_bsidx],
                encoder_attention_mask=image_atts[entis_bsidx],
                return_dict=True,
                mode='fusion'
            )

            entis_vl_embeddings = torch.cat(
                [
                    output_pos_ts.last_hidden_state[:, 0, :],
                    output_pos_is.last_hidden_state[:, 0, :],
                    output_neg_entis.last_hidden_state[:, 0, :], 
                ],
                dim=0,
            )
            entis_data_samples = [DataSample(is_matched=True) for _ in range(2 * bs)] + \
                                    [DataSample(is_matched=False) for _ in range(2 * bs)]

            entis_itm_loss = self.itm_head.loss((entis_vl_embeddings, ), entis_data_samples)

            loss_multimodal.update({
                "entis_itm_loss" : entis_itm_loss['itm_loss'] * self.w_itm_entis,
                "entis_itm_accuracy" : entis_itm_loss['itm_accuracy']
            })
        
        if total_bb_loss:
            return dict(ChainMap(loss, loss_multimodal, 
                                 {"bboxes_loss":total_bb_loss}))
        else:
            return dict(ChainMap(loss, loss_multimodal))
        
    def predict(self, images, data_samples, cal_i2t=True, cal_t2i=True):
        feats = self.extract_feat(images, data_samples)

        return self.predict_all(
            feats, data_samples, cal_i2t=cal_i2t, cal_t2i=cal_t2i)

    def predict_all(self,
                    feats,
                    data_samples,
                    num_images=None,
                    num_texts=None,
                    cal_i2t=True,
                    cal_t2i=True):

        text_attn_mask = feats['text_attn_mask']
        image_embeds = feats.get('image_embeds', None)
        image_feat = feats['image_feat']
        text_embeds = feats.get('text_embeds', None)
        text_feat = feats['text_feat']

        num_images = num_images or image_feat.size(0)
        num_texts = num_texts or text_feat.size(0)

        results = []

        if cal_i2t:
            result_i2t = self.compute_score_matrix_i2t(
                image_feat,
                image_embeds,
                all_gather_concat(text_feat)[:num_texts],
                all_gather_concat(text_embeds)[:num_texts] if not self.fast_match else None,
                all_gather_concat(text_attn_mask)[:num_texts],
            )
            results.append(
                self._get_predictions(result_i2t, data_samples, mode='i2t'))

        if cal_t2i:
            result_t2i = self.compute_score_matrix_t2i(
                all_gather_concat(image_feat)[:num_images],
                all_gather_concat(image_embeds)[:num_images] if not self.fast_match else None,
                text_feat,
                text_embeds,
                text_attn_mask,
            )
            results.append(
                self._get_predictions(result_t2i, data_samples, mode='t2i'))
        return tuple(results)
    

    def _rate_ensemble(self, score, cur_topk_sim, n=50):
        device = score.device
        rate_values = torch.linspace(0.9, 0.99, steps=n, device=device)
        ensemble_list = []
        
        for r in rate_values:
            temp_out = r * F.softmax(score, dim=-1) + (1 - r) * cur_topk_sim
            ensemble_list.append(temp_out)
        ensemble_tensor = torch.stack(ensemble_list, dim=0)
        
        mean_val = torch.mean(ensemble_tensor, dim=0, keepdim=True)
        diff = -torch.abs(ensemble_tensor - mean_val)
        weights = F.softmax(diff, dim=0)
        outs = torch.sum(weights * ensemble_tensor, dim=0)
        return outs
    

    def compute_score_matrix_t2i(self, img_feats, img_embeds, text_feats, text_embeds, text_attn_mask,
                                load_batch=512, use_fp16=True):

        if self.fast_match:
            score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                    -100.0, device="cpu", dtype=torch.float32)
            num_batches = (text_feats.size(0) + load_batch - 1) // load_batch
            for i in range(num_batches):
                start_idx = i * load_batch
                end_idx = min((i + 1) * load_batch, text_feats.size(0))

                text_feats_batch = text_feats[start_idx:end_idx].to(self.device)
                sim_matrix_batch = text_feats_batch @ img_feats.t().to(self.device)
                score_matrix_t2i[start_idx:end_idx] = sim_matrix_batch.cpu()

            return score_matrix_t2i
        
        encoder_batch = (load_batch // self.topk) * 2
        score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                    -100.0, device="cpu", dtype=torch.float32)
        
        num_batches = (text_feats.size(0) + load_batch - 1) // load_batch
        for i in range(num_batches):
            start_idx = i * load_batch
            end_idx = min((i + 1) * load_batch, text_feats.size(0))
            
            text_feats_batch = text_feats[start_idx:end_idx].to(self.device)
            text_embeds_batch = text_embeds[start_idx:end_idx].to(self.device)
            text_attn_mask_batch = text_attn_mask[start_idx:end_idx].to(self.device)
            
            with autocast(enabled=use_fp16):
                sim_matrix_batch = text_feats_batch @ img_feats.t().to(self.device)
                
                topk_sim, topk_idx = sim_matrix_batch.topk(k=self.topk, dim=1)
                
                cur_batch_size = end_idx - start_idx
                
                num_inner_batches = (cur_batch_size + encoder_batch - 1) // encoder_batch
                for j in range(num_inner_batches):
                    inner_start = j * encoder_batch
                    inner_end = min((j + 1) * encoder_batch, cur_batch_size)
                    inner_batch_size = inner_end - inner_start
                    
                    cur_text_embeds = text_embeds_batch[inner_start:inner_end]
                    cur_text_attn_mask = text_attn_mask_batch[inner_start:inner_end]
                    cur_topk_idx = topk_idx[inner_start:inner_end]
                    cur_topk_sim = topk_sim[inner_start:inner_end]
                    
                    topk_idx_flat = cur_topk_idx.reshape(-1).cpu()
                    encoder_output = img_embeds[topk_idx_flat].to(self.device)
                    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                    
                    text_embeds_expand = cur_text_embeds.unsqueeze(1).expand(-1, self.topk, -1, -1)
                    text_embeds_expand = text_embeds_expand.reshape(-1, cur_text_embeds.size(1), cur_text_embeds.size(2))
                    text_attn_mask_expand = cur_text_attn_mask.unsqueeze(1).expand(-1, self.topk, -1)
                    text_attn_mask_expand = text_attn_mask_expand.reshape(-1, cur_text_attn_mask.size(1))
                    
                    output = self.text_encoder(
                        encoder_embeds=text_embeds_expand,
                        attention_mask=text_attn_mask_expand,
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode='fusion'
                    )
                    
                    score = self.itm_head((output.last_hidden_state[:, 0, :],))[:, 1]
                    score = score.reshape(inner_batch_size, self.topk)
                    
                    batch_indices = torch.arange(start_idx + inner_start, start_idx + inner_end, device="cpu")
                    rows = batch_indices.unsqueeze(1).expand(-1, self.topk)
                    outs = self._rate_ensemble(score, cur_topk_sim)
                    # outs = score + cur_topk_sim
                    # outs = score
                    if use_fp16:
                        outs = outs.to(torch.float32)
                    score_matrix_t2i[rows, cur_topk_idx.cpu()] = outs.cpu()
                    
        return score_matrix_t2i

    def compute_score_matrix_i2t(self, img_feats, img_embeds, text_feats, text_embeds, text_attn_mask,
                                load_batch=256, use_fp16=True):
        if self.fast_match:
            score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                    -100.0, device="cpu", dtype=torch.float32)
            num_batches = (img_feats.size(0) + load_batch - 1) // load_batch

            for i in track_on_main_process(range(num_batches), 'Compute I2T scores...', total=num_batches):
                start_idx = i * load_batch
                end_idx = min((i + 1) * load_batch, img_feats.size(0))
                
                img_feats_batch = img_feats[start_idx:end_idx].to(self.device)
                sim_matrix_batch = img_feats_batch @ text_feats.t().to(self.device)
                score_matrix_i2t[start_idx:end_idx] = sim_matrix_batch.cpu()

            return score_matrix_i2t
        
        encoder_batch = load_batch // self.topk
        score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                    -100.0, device="cpu", dtype=torch.float32)
        
        num_batches = (img_feats.size(0) + load_batch - 1) // load_batch
        for i in track_on_main_process(range(num_batches), 'Compute I2T scores...', total=num_batches):
            start_idx = i * load_batch
            end_idx = min((i + 1) * load_batch, img_feats.size(0))
            
            img_feats_batch = img_feats[start_idx:end_idx].to(self.device)
            img_embeds_batch = img_embeds[start_idx:end_idx].to(self.device)
            
            with autocast(enabled=use_fp16):
                sim_matrix_batch = img_feats_batch @ text_feats.t().to(self.device)
                
                topk_sim, topk_idx = sim_matrix_batch.topk(k=self.topk, dim=1)
                
                cur_batch_size = end_idx - start_idx
                
                num_inner_batches = (cur_batch_size + encoder_batch - 1) // encoder_batch
                for j in range(num_inner_batches):
                    inner_start = j * encoder_batch
                    inner_end = min((j + 1) * encoder_batch, cur_batch_size)
                    inner_batch_size = inner_end - inner_start
                    
                    cur_img_embeds = img_embeds_batch[inner_start:inner_end]
                    cur_topk_idx = topk_idx[inner_start:inner_end]
                    cur_topk_sim = topk_sim[inner_start:inner_end]
                    
                    topk_idx_flat = cur_topk_idx.reshape(-1).cpu()
                    text_embeds_batch = text_embeds[topk_idx_flat].to(self.device)
                    text_attn_mask_batch = text_attn_mask[topk_idx_flat].to(self.device)
                    
                    img_embeds_expand = cur_img_embeds.unsqueeze(1).expand(-1, self.topk, -1, -1)
                    img_embeds_expand = img_embeds_expand.reshape(-1, cur_img_embeds.size(1), cur_img_embeds.size(2))
                    encoder_att = torch.ones(img_embeds_expand.size()[:-1], dtype=torch.long).to(self.device)
                    
                    output = self.text_encoder(
                        encoder_embeds=text_embeds_batch,
                        attention_mask=text_attn_mask_batch,
                        encoder_hidden_states=img_embeds_expand,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode='fusion'
                    )
                    
                    score = self.itm_head((output.last_hidden_state[:, 0, :],))[:, 1]
                    score = score.reshape(inner_batch_size, self.topk)
                    batch_indices = torch.arange(start_idx + inner_start, start_idx + inner_end, device="cpu")
                    rows = batch_indices.unsqueeze(1).expand(-1, self.topk)
                    outs = self._rate_ensemble(score, cur_topk_sim)
                    # outs = score + cur_topk_sim
                    # outs = score
                    if use_fp16:
                        outs = outs.to(torch.float32)
                    score_matrix_i2t[rows, cur_topk_idx.cpu()] = outs.cpu()
                    
        return score_matrix_i2t

    def _get_predictions(self,
                         result: torch.Tensor,
                         data_samples: List[DataSample],
                         mode: str = 'i2t'):

        # create data sample if not exists
        if data_samples is None:
            data_samples = [DataSample() for _ in range(result.size(0))]
        elif mode == 't2i':
            # Process data samples to align with the num of texts.
            new_data_samples = []
            for sample in data_samples:
                if isinstance(sample.text, (list, tuple)):
                    texts = sample.text
                else:
                    texts = [sample.text]
                for i, text in enumerate(texts):
                    new_sample = DataSample(text=text)
                    if 'gt_image_id' in sample:
                        new_sample.gt_label = sample.gt_image_id[i]
                    new_data_samples.append(new_sample)
            assert len(new_data_samples) == result.size(0)
            data_samples = new_data_samples
        elif mode == 'i2t':
            for sample in data_samples:
                if 'gt_text_id' in sample:
                    sample.gt_label = sample.gt_text_id
        else:
            raise ValueError(f'Type {mode} is not supported.')

        for data_sample, score in zip(data_samples, result):
            idx = score.argmax(keepdim=True).detach()
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(idx)
        return data_samples

    # TODO: add temperaily
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for (name,
                 param), (name_m,
                          param_m) in zip(model_pair[0].named_parameters(),
                                          model_pair[1].named_parameters()):
                # hack to behave the same
                if any([i in name for i in ['8', '9', '10', '11']
                        ]) and 'layers' in name and any(
                            [i in name for i in ['attn', 'ffn']]):
                    param_m.data = param.data
                else:
                    param_m.data = param_m.data * self.momentum + \
                        param.data * (1.0 - self.momentum)
