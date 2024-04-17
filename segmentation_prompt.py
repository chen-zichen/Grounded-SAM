'''
    Description: This file contains the code for the segmentation with prompt input.
'''

import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import argparse
import os
import json

parser = argparse.ArgumentParser(description="Segmentation with prompt input")
parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
parser.add_argument("--dino_path", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to the grounding dino model")
parser.add_argument("--dino_checkpoint", type=str, default="./groundingdino_swint_ogc.pth", help="Path to the grounding dino checkpoint")

parser.add_argument("--sam_version", type=str, default="vit_h", help="Version of the SAM model")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_h_4b8939.pth", help="Path to the SAM checkpoint")

parser.add_argument("--image_path", type=str, default="./assets/demo2.jpg", help="Path to the image")
parser.add_argument("--prompt_path", type=str, default="./", help="Classes here is the prompt.")
parser.add_argument("--box_threshold", type=float, default=0.25, help="Box threshold")
parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
parser.add_argument("--nms_threshold", type=float, default=0.8, help="NMS threshold")
parser.add_argument("--dataset_name", type=str, default="Clipart", help="Name of the dataset")
parser.add_argument("--dataset_path", type=str, default="./data/", help="Path to the dataset")

arg = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = arg.dino_path
GROUNDING_DINO_CHECKPOINT_PATH = arg.dino_checkpoint

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = arg.sam_version
SAM_CHECKPOINT_PATH = arg.sam_checkpoint

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = arg.image_path
CLASSES_PATH = arg.prompt_path
CLASSES = []   # TODO
BOX_THRESHOLD = arg.box_threshold
TEXT_THRESHOLD = arg.text_threshold
NMS_THRESHOLD = arg.nms_threshold



# load image
DATASET = arg.dataset_name
DATASET_PATH = arg.dataset_path
dataset_path = f"{DATASET_PATH}/{DATASET}/"

# load data instance
data_instance = f"{dataset_path}/data/prompts_selected.jsonl"

for line in open(data_instance):
    line = json.loads(line)

    # TODO: new caption
    SOURCE_IMAGE_PATH = line["image"]
    CLASSES = line["prompt"]

    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    output_path = f"{dataset_path}/output/dino"
    # check if the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    task_id = line["task_id"]

    # save the annotated grounding dino image
    cv2.imwrite(f"{output_path}/{task_id}.jpg", annotated_frame)


    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    output_path = f"{dataset_path}/output/grounded_sam"
    # check if the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    cv2.imwrite(f"{output_path}/{task_id}.jpg", annotated_image)
