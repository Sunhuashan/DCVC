CUDA_VISIBLE_DEVICES='0' python -u ./train_image.py -d /root/autodl-tmp/ \
    --cuda --epochs 50 --lr_epoch 45 48 --save      \
    --save_path /root/DCVC/DCVC-HEM/checkpoints/