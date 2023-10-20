_base_ = '../ViTDet/cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_config = dict(interval=1)
evaluation = dict(interval=12, metric=['bbox', 'segm'])
log_config = dict(interval=200, hooks=[dict(type='TextLoggerHook')])

checkpoint_file = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

model = dict(
    backbone=dict(pretrained=checkpoint_file)
    )

data = dict(samples_per_gpu=4)

optimizer = dict(lr=0.0002)