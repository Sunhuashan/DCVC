python test_video.py \
        --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
        --force_intra True \
        --rate_num 4 --test_config ./dataset_config_example.json \
        --cuda 1 -w 1 --write_stream 0 --output_path output.json

# ROOT="/root/autodl-tmp/test/"

# DATASET="kodak"
# # DATASET="ticnick"
# # DATASET="jepgai"
# # DATASET="clic21_test"
# # DATASET="clic_pro_val"
# # DATASET="clic22_test"
# DATA_PATH="${ROOT}${DATASET}"


# python3 eval.py --checkpoint "/root/DCVC/DCVC-HEM/checkpoints/acmmm2022_image_psnr.pth.tar" \
#                 --data "$DATA_PATH" --cuda