import sys
import os
from pathlib import Path

# 홈 디렉토리의 YOLOv5 경로 추가
HOME = Path.home()
yolov5_path = HOME / "yolov5"
sys.path.append(str(yolov5_path))

import torch
import cv2
import numpy as np

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# 모델 로드
weights = HOME / "Documents/vscode/github_traffic-sign/kaggle_dataset/yolo_data/result_weights/best.pt"
device = select_device('')  # GPU 사용 시 'cuda:0', CPU 사용 시 ''
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt

# 이미지 크기 설정
imgsz = check_img_size((640, 640), s=stride)  # 모델 stride에 맞게 이미지 크기 조정

# 이미지 로드 및 전처리
img_path = HOME / "Documents/vscode/github_traffic-sign/kaggle_dataset/yolo_data/1.png"
img0 = cv2.imread(str(img_path))
img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]  # 이미지 리사이즈 및 패딩
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device)
img = img.float()  # uint8 to fp16/32
img /= 255  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim

# 추론 실행
pred = model(img)
pred = non_max_suppression(pred)[0]

# 결과 처리 및 시각화
pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
annotator = Annotator(img0, line_width=3, example=str(names))
for *xyxy, conf, cls in reversed(pred):
    c = int(cls)
    label = f'{names[c]} {conf:.2f}'
    annotator.box_label(xyxy, label, color=colors(c, True))

# 결과 저장
output_path = HOME / "Documents/vscode/github_traffic-sign/kaggle_dataset/yolo_data/results.jpg"
cv2.imwrite(str(output_path), img0)

# 결과 출력
for *xyxy, conf, cls in pred:
    print(f"Class: {names[int(cls)]}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

print(f"Detection completed. Results saved as {output_path}")