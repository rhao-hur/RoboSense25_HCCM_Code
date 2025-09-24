_base_ = [
    '../_base_/datasets/geotext1652_retrieval.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='XVLMRetrieval_hccm',
    init_cfg = dict(
            type='Pretrained', 
            checkpoint = "pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth",
    ),
    tokenizer_path='pretrain/bert-base-uncased',
    vision_encoder=dict(
        type='XVLM_SwinTransformer',
        img_size=384,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[ 2, 2, 18, 2 ],
        num_heads=[ 4, 8, 16, 32 ],
        window_size=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    ),
    text_encoder=dict(
        type='XVLM_XBert',
        med_config=dict(
            architectures=['BertForMaskedLM'], 
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            type_vocab_size=2,
            vocab_size=30522,
            fusion_layer=6,
            encoder_width=1024,
            add_cross_attention=True),
    ),
    vision_proj=dict(
        type='Linear',
        in_features=1024,
        out_features=256,
    ),
    text_proj=dict(
        type='Linear',
        in_features=768,
        out_features=256,
    ),
    itm_head=dict(
        type='XVLM_ITMHead',
        hidden_size=768,
        with_pooler=False,
        cal_acc=True
    ),
    itc_head=dict(
        type='XVLM_ITC_MCD',
        embed_dim=256,
        queue_size=57600,
        use_distill=True,
        alpha=0.4,
    ),
    bbox_head=dict(
        type='XVLM_BOXHead',
        hidden_size=768,
    ),
    
    topk=256,
    train_max_words=90,
    val_max_words=90,
    max_tokens=90,

    w_itc=0.25,
    w_itc_entis=0.25,
    w_itm=1,
    w_itm_entis=0.5,

    fast_match=False
)

# dataset
train_dataloader = dict(batch_size=24)
val_dataloader = dict(batch_size=64)
test_dataloader = dict(batch_size=64)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=dict(
        type='AdamW', lr=3e-5, weight_decay=0.01
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.,
    )
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        end_factor=1,
        end=1000, 
        by_epoch=False
    ),
    dict(
        type='LinearLR', 
        start_factor=1, 
        end_factor=1e-10, 
        by_epoch=False, 
        begin=1000)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

val_cfg = dict(
    type='HDCRetrievalValLoop',
    fp16=True,
    load_cpu = True,
    fast_datainfo = True,
    i2t = False,
)
test_cfg = dict(
    type='HDCRetrievalTestLoop', 
    fp16=True,
    load_cpu = True,
    fast_datainfo = True,
    i2t = True,
)

randomness = dict(seed=1)

default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(by_epoch=True, interval=1),
)