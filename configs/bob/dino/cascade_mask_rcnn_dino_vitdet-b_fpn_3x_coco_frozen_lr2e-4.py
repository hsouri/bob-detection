_base_ = '../ViTDet/cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'

model = dict(
    backbone=dict(
        frozen_stages=12, # For ViTdet, stages are same as layers
        pretrained=checkpoint_file,
    ),
)

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.0002)