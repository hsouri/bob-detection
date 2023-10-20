_base_ = './cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTdet',
        img_size=1024,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_emb=True,
        frozen_stages=12, # For ViTdet, stages are same as layers
        pretrained=checkpoint_file,
    ),
    neck=dict(in_channels=[384, 384, 384, 384]),
)

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.0001)