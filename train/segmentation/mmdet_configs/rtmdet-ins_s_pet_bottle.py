# RTMDet-Ins-s を pet_bottle 3クラス（bottle/cap/label）でファインチューンする設定。
# mmdetection v3.3.0 / mmcv 2.1.0 / torch 2.1.0 cu121（RunPod セットアップは
# runpod/setup_train.sh、実行は runpod/run_train.sh）。
#
# - データ: instances_{train,val,test}_trainready.json（_sam3full ベース、
#   depicted は iscrowd=1 で学習から除外済み。file_name は images/all/ 込みなので
#   data_prefix は空）
# - COCO 事前学習 (rtmdet-ins_s 300e) から 60 epoch、単一 GPU バッチ32、
#   lr は線形スケール（0.004 × 32/256 = 5e-4）
_base_ = '/workspace/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'

data_root = '/workspace/pet_bottle/'
metainfo = dict(
    classes=('bottle', 'cap', 'label'),
    palette=[(220, 20, 60), (0, 120, 255), (0, 200, 80)],
)

num_classes = 3
max_epochs = 60
stage2_switch_epoch = 50   # 最後の10epochは重い増強(Mosaic/MixUp)をオフ
base_lr = 0.0005
train_batch = 32

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=train_batch,
    num_workers=10,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train_trainready.json',
        data_prefix=dict(img=''),
    ),
)
val_dataloader = dict(
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val_trainready.json',
        data_prefix=dict(img=''),
    ),
)
test_dataloader = dict(
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test_trainready.json',
        data_prefix=dict(img=''),
    ),
)
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val_trainready.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test_trainready.json')

optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2,
         by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(max_epochs=max_epochs, val_interval=5,
                 dynamic_intervals=[(stage2_switch_epoch, 1)])

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, save_best='coco/segm_mAP'),
)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook', switch_epoch=stage2_switch_epoch,
         switch_pipeline={{_base_.train_pipeline_stage2}}),
]

# MLflow 記録（file ストア、サーバ不要）。/workspace/mlruns を成果物と一緒に持ち帰り、
# ローカルで `mlflow ui --backend-store-uri file:<repo>/train/segmentation/mlruns` で閲覧。
# run 名は実行時に --cfg-options で上書き可:
#   --cfg-options visualizer.vis_backends.1.run_name=h100x8_60e
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend',
             save_dir='/workspace/mlruns',
             exp_name='pet_bottle_rtmdet_ins',
             tags=dict(dataset='_sam3full+trainready 2026-07-11',
                       model='rtmdet-ins_s', pretrain='coco_300e')),
    ])

# COCO 事前学習チェックポイント（run_train.sh が /workspace/checkpoints に取得）
load_from = ('/workspace/checkpoints/rtmdet-ins_s_coco.pth')
