_base_ = '../ViTDet/cascade_mask_rcnn_vitdet-b_fpn_3x_coco_base.py'

checkpoint_file = 'https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin'

model = dict(
    backbone=dict(
        pre_norm=True,
        frozen_stages=12, # For ViTdet, stages are same as layers
        pretrained=checkpoint_file
    )
)

data = dict(samples_per_gpu=2)

optimizer = dict(lr=0.00008)