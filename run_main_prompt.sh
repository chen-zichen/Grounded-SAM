export CUDA_VISIBLE_DEVICES=0,1
# videoname is a list

# VIDEO_NAME=("bear")
# VIDEO_NAME=("bear" "car" "car_turn" "cat_flower" "dog_walking" "fox" "rabbit_strawberry" "rabbit_watermelon" "squirrel_carrot" "swan")
VIDEO_NAME=("car" "rabbit_watermelon")
for i in "${VIDEO_NAME[@]}"
do
  echo "Processing video $i"
  python segmentation_prompt.py \
    --video_name $i
done

# python segmentation.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --ram_checkpoint ram_swin_large_14m.pth \
#   --grounded_checkpoint groundingdino_swint_ogc.pth \
#   --sam_checkpoint sam_vit_h_4b8939.pth \
#   --output_dir "outputs" \
#   --box_threshold 0.20 \
#   --text_threshold 0.20 \
#   --iou_threshold 0.5 \
#   --device "cuda" \
#   --dataset_name "video" \