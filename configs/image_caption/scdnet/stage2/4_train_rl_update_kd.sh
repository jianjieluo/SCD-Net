python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage2/diffusion_rl.yaml \
    OUTPUT_DIR output/stage2/rl_update_kd \
    DATALOADER.CASCADED_FILE output/stage1/rl/ep42_inference_train/results/ep_42_ts_20_td_0.pkl \
    MODEL.WEIGHTS output/stage2/rl/model_Epoch_00060_Iter_0106199.pth \
    DATALOADER.FORCE_GUIDED False \
    DATALOADER.KD_PRED_FILE output/stage2/rl/ep60_inference_train/results/merge_ep_60_ts_20_td_0_with_ar_teacher_pred_ep25.pkl