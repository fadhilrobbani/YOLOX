#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import time
import cv2
import numpy as np
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, vis

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Webcam Inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model file",
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
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID (usually 0 for built-in camera)",
    )
    return parser

def main():
    args = make_parser().parse_args()
    
    # Initialize ONNX runtime session
    session = onnxruntime.InferenceSession(args.model)
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    print("Starting live camera inference. Press 'q' to quit...")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Preprocess frame
            img, ratio = preprocess(frame, input_shape)
            
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
            
            # Visualize results
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                frame = vis(
                    frame, 
                    final_boxes, 
                    final_scores, 
                    final_cls_inds,
                    conf=args.score_thr, 
                    class_names=args.classes
                )
            
            # Display the resulting frame
            cv2.imshow('YOLOX Live Detection', frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Add small delay to control frame rate
            time.sleep(0.01)
                
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()