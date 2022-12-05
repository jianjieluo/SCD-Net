python test_net_ema_only.py --num-gpus 1 \
    --config-file configs/image_caption/scdnet/stage1/diffusion_inf_train.yaml \
    OUTPUT_DIR output/stage1/xe/ep48_inference_train \
    MODEL.WEIGHTS output/stage1/xe/model_Epoch_00048_Iter_0084959.pth