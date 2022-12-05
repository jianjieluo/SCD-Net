python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage2/diffusion_rl.yaml \
    OUTPUT_DIR output/stage2/rl \
    DATALOADER.CASCADED_FILE output/stage1/rl/ep42_inference_train/results/ep_42_ts_20_td_0.pkl \
    MODEL.WEIGHTS output/stage2/xe/model_Epoch_00056_Iter_0099119.pth \
    DATALOADER.FORCE_GUIDED False