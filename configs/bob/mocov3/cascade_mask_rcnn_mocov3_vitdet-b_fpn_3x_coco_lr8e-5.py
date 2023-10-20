_base_ = '../ViTDet/cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'

model = dict(
    backbone=dict(pretrained=checkpoint_file)
    )

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.00008)