python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage1/diffusion_rl.yaml \
    OUTPUT_DIR output/stage1/rl \
    MODEL.WEIGHTS output/stage1/xe/model_Epoch_00048_Iter_0084959.pth