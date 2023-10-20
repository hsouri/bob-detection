_base_ = './cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'

model = dict(
    backbone=dict(pretrained=checkpoint_file)
    )

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.0002)