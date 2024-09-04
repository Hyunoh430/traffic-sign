import sys
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

def inference_and_save(model, input_image_path, output_image_path, threshold=0.3, class_names=None):
    # 이미지 로드 및 전처리
    img = Image.open(input_image_path).convert("L")  # 그레이스케일로 변환
    img = img.resize((128, 128))  # 이미지 크기 조정
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # 이미지 텐서로 변환 및 배치 차원 추가

    # 모델 추론 시간 측정 시작
    start_time = time.time()

    # 모델 추론
    with torch.no_grad():
        predictions = model(img_tensor.to(torch.device('cpu')))  # CPU로 모델 실행

    # 추론 시간 계산
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # 결과 시각화
    fig, ax = plt.subplots(1, figsize=(1.28, 1.28), dpi=100)  # 128x128 픽셀로 설정
    img_np = np.array(img)
    ax.imshow(img_np, cmap='gray')

    for i, (box, score, label) in enumerate(zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels'])):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            if class_names:
                class_name = class_names[label.item()]
            else:
                class_name = f"Class {label.item()}"
            ax.text(x1, y1, f'{class_name}', fontsize=4, bbox=dict(facecolor='white', alpha=0.5, linewidth=0))

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 여백 제거
    plt.savefig(output_image_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Result saved at {output_image_path}')

# 사용 예시
input_image_path = sys.argv[1]  # 입력 이미지 경로
output_image_path = sys.argv[2]  # 출력 이미지 저장 경로

# 모델 로드
model_path = '/home/Pi/Documents/traffic/gray.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# 클래스 이름 정의 (예시)
class_names = {0: 'trafficlight', 1: 'speedlimit', 2: 'crosswalk', 3: 'stop'}

# 추론 및 저장
inference_and_save(model, input_image_path, output_image_path, threshold=0.3, class_names=class_names)