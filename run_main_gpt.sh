export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=https://closeai.deno.dev/v1
export CUDA_VISIBLE_DEVICES=0
python automatic_label_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo0.jpg \
  --output_dir "outputs" \
  --box_threshold 0.15 \
  --text_threshold 0.15 \
  --iou_threshold 0.5 \
  --device "cuda"