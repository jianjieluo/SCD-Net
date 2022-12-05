CUDA_VISIBLE_DEVICES=0 python test_net_ema_only.py --num-gpus 1 \
    --config-file configs/image_caption/scdnet/stage1/diffusion_rl_inf_train.yaml \
    MODEL.WEIGHTS output/stage1/rl/model_Epoch_00042_Iter_0074339.pth \
    OUTPUT_DIR output/stage1/rl/ep42_inference_train