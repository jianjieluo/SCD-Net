CUDA_VISIBLE_DEVICES=0 python test_net_ema_only.py --num-gpus 1 \
    --config-file configs/image_caption/scdnet/stage2/diffusion_rl_inf_train.yaml \
    DATALOADER.CASCADED_FILE output/stage1/rl/ep42_inference_train/results/ep_42_ts_20_td_0.pkl \
    MODEL.WEIGHTS output/stage2/rl/model_Epoch_00060_Iter_0106199.pth \
    OUTPUT_DIR output/stage2/rl/ep60_inference_train
