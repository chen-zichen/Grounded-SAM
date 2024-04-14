export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo6.jpg \
  --output_dir "outputs" \
  --box_threshold 0.15 \
  --text_threshold 0.15 \
  --iou_threshold 0.5 \
  --device "cuda"