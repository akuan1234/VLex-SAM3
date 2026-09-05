_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/prompt_banks/vaihingen_vlex_sam3.txt',
    prob_thd=0.0,
    bg_idx=5,
    confidence_threshold=0.5,
    evaluation_class_names=[
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ],
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = 'data/Vaihingen'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # Raw Vaihingen masks use 0=ignore and 1..6=semantic classes.
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
