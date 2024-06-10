# table 1 scd_net xe
# Example: you can use 'test_net_cas.py' for cascaded inference online
CUDA_VISIBLE_DEVICES=0 python test_net_cas.py --num-gpus 1 \
    --config-file configs/image_caption/scdnet/stage2/cascaded_diffusion_inference.yaml \
    OUTPUT_DIR final_performance/table1/scd_net_xe

# table 1 scd_net rl
# Example: you can also cache the stage1 inference output offline and then run the cascaded stage2 inference
CUDA_VISIBLE_DEVICES=0 python test_net_ema_only.py --num-gpus 1 \
    --config-file configs/image_caption/scdnet/stage2/diffusion_rl.yaml \
    OUTPUT_DIR final_performance/table1/scd_net_rl_ep55 \
    MODEL.WEIGHTS models/stage2/rl/model_Epoch_00055_Iter_0097349.pth \
    DATALOADER.CASCADED_FILE models/stage1/rl/ep_30_ts_20_td_0.pkl \
    SEED 1234
