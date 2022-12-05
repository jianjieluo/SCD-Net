python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage2/diffusion.yaml \
    OUTPUT_DIR output/stage2/xe \
    DATALOADER.CASCADED_FILE output/stage1/xe/ep48_inference_train/results/ep_48_ts_50_td_0.pkl