_base_ = './cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

model = dict(
    backbone=dict(pretrained=None)
    )

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.0002)