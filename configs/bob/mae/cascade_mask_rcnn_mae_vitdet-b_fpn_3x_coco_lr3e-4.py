_base_ = '../ViTDet/cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'

model = dict(
    backbone=dict(pretrained=checkpoint_file)
    )

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.0003)