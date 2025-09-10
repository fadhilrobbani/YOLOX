#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import numpy as np
import onnxruntime
from glob import glob
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, vis

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Image Inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="onnx_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "-c",
        "--classes",
        nargs='+',
        default=["cup", "cup_body"],
        help="List of class names",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Model input shape (width,height)",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    return parser

def main():
    args = make_parser().parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize ONNX runtime session
    session = onnxruntime.InferenceSession(args.model)
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Get image paths
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob(os.path.join(args.image_dir, f"*{ext}")))
    
    for image_path in image_paths:
        # Read and preprocess image
        origin_img = cv2.imread(image_path)
        img, ratio = preprocess(origin_img, input_shape)
        
        # Run inference
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        
        # Process output
        predictions = demo_postprocess(output[0], input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        # Convert boxes to xyxy format
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        
        # Apply NMS
        dets = multiclass_nms(
            boxes_xyxy, 
            scores, 
            nms_thr=0.45, 
            score_thr=0.1
        )
        
        # Visualize and save results
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result_img = vis(
                origin_img, 
                final_boxes, 
                final_scores, 
                final_cls_inds,
                conf=args.score_thr, 
                class_names=args.classes
            )
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, result_img)
            print(f"Saved result to {output_path}")

if __name__ == '__main__':
    main()