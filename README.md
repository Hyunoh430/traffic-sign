# YOLOv5 Grayscale Sign Detection

This project implements a custom YOLOv5 model for detecting signs in 64x64 grayscale images.

## Setup

1. Clone the YOLOv5 repository:
   ```
   !git clone https://github.com/ultralytics/yolov5
   %cd yolov5
   ```

2. Install the required dependencies (assuming you're using a Colab environment with PyTorch pre-installed).

## Data Preparation

1. Create a function to split the dataset into train and validation sets:

```python
import os
import shutil
import random

def split_data(source_img_dir, source_label_dir, train_dir, val_dir, split_ratio=0.2):
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    
    all_files = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    num_val = int(len(all_files) * split_ratio)
    val_files = random.sample(all_files, num_val)
    
    for file in all_files:
        img_src = os.path.join(source_img_dir, file)
        label_src = os.path.join(source_label_dir, os.path.splitext(file)[0] + '.txt')
        
        if file in val_files:
            img_dst = os.path.join(val_dir, 'images', file)
            label_dst = os.path.join(val_dir, 'labels', os.path.splitext(file)[0] + '.txt')
        else:
            img_dst = os.path.join(train_dir, 'images', file)
            label_dst = os.path.join(train_dir, 'labels', os.path.splitext(file)[0] + '.txt')
        
        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
```

2. Execute the data split:

```python
# 데이터 분할 실행
source_img_dir = '/content/drive/MyDrive/Colab_Notebooks/traffic-yolo/yolo_data/images' # 원본 이미지가 있는 디렉토리
source_label_dir = '/content/drive/MyDrive/Colab_Notebooks/traffic-yolo/yolo_data/labels' # 원본 라벨이 있는 디렉토리
train_dir = '/content/dataset/train'
val_dir = '/content/dataset/valid'
split_data(source_img_dir, source_label_dir, train_dir, val_dir)
```

3. Process the images (convert to grayscale and resize):

```python
import cv2
import numpy as np

def process_images(input_dir, output_dir, size=(64, 64)):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, size)
            cv2.imwrite(os.path.join(output_dir, filename), resized)

# 훈련 및 검증 이미지 처리
process_images(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'images'))
process_images(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'images'))
```

## Configuration

1. Create a custom data YAML file:

```python
import yaml

custom_data_yaml = {
    'train': '/content/drive/MyDrive/yolo_gray_dataset/train/images',
    'val': '/content/drive/MyDrive/yolo_gray_dataset/valid/images',
    'nc': 1,
    'names': ['sign'],
    'img_size': 64,
    'channels': 1
}

yaml_path = '/content/yolov5/data/custom_data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(custom_data_yaml, f, default_flow_style=False)
print(f"YAML file created at: {yaml_path}")
```

2. Create a modified YOLOv5n model configuration for 64x64 grayscale input:

```python
model_yaml = """
# YOLOv5n modified for 64x64 grayscale
nc: 1 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.25 # layer channel multiple
anchors:
 - [4,5, 8,10, 13,16] # P3/8
 - [23,29, 43,55, 73,105] # P4/16
backbone:
 [[-1, 1, Conv, [32, 3, 2, 1]], # 0-P1/2
 [-1, 1, Conv, [64, 3, 2]], # 1-P2/4
 [-1, 3, C3, [64]],
 [-1, 1, Conv, [128, 3, 2]], # 3-P3/8
 [-1, 3, C3, [128]],
 [-1, 1, Conv, [256, 3, 2]], # 5-P4/16
 [-1, 3, C3, [256]],
 [-1, 1, SPPF, [256, 5]], # 7
 ]
head:
 [[-1, 1, Conv, [128, 1, 1]],
 [-1, 1, nn.Upsample, [None, 2, 'nearest']],
 [[-1, 4], 1, Concat, [1]], # cat backbone P3
 [-1, 3, C3, [128, False]], # 11
 [-1, 1, Conv, [128, 3, 2]],
 [[-1, 6], 1, Concat, [1]], # cat head P4
 [-1, 3, C3, [256, False]], # 14 (P4/16)
 [[11, 14], 1, Detect, [nc, anchors]], # Detect(P3, P4)
 ]
"""

with open('/content/yolov5/models/yolov5n-gray.yaml', 'w') as f:
    f.write(model_yaml)
print("Modified YOLOv5n model configuration for 64x64 grayscale input saved.")
```

## Training

Run the training script:

```
!python train.py --img 64 --batch 64 --epochs 300 --data /content/yolov5/data/custom_data.yaml --cfg /content/yolov5/models/yolov5n-gray.yaml --weights '' --name yolov5n_gray_sign --cache
```

This command will train the model for 300 epochs using the custom configuration and dataset.

## Complete Code

Here's the complete code for the entire process:

```python
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
%cd yolov5

import os
import shutil
import random
import cv2
import numpy as np
import yaml

def split_data(source_img_dir, source_label_dir, train_dir, val_dir, split_ratio=0.2):
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    
    all_files = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    num_val = int(len(all_files) * split_ratio)
    val_files = random.sample(all_files, num_val)
    
    for file in all_files:
        img_src = os.path.join(source_img_dir, file)
        label_src = os.path.join(source_label_dir, os.path.splitext(file)[0] + '.txt')
        
        if file in val_files:
            img_dst = os.path.join(val_dir, 'images', file)
            label_dst = os.path.join(val_dir, 'labels', os.path.splitext(file)[0] + '.txt')
        else:
            img_dst = os.path.join(train_dir, 'images', file)
            label_dst = os.path.join(train_dir, 'labels', os.path.splitext(file)[0] + '.txt')
        
        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

# 데이터 분할 실행
source_img_dir = '/content/drive/MyDrive/Colab_Notebooks/traffic-yolo/yolo_data/images'
source_label_dir = '/content/drive/MyDrive/Colab_Notebooks/traffic-yolo/yolo_data/labels'
train_dir = '/content/dataset/train'
val_dir = '/content/dataset/valid'
split_data(source_img_dir, source_label_dir, train_dir, val_dir)

def process_images(input_dir, output_dir, size=(64, 64)):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, size)
            cv2.imwrite(os.path.join(output_dir, filename), resized)

# 훈련 및 검증 이미지 처리
process_images(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'images'))
process_images(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'images'))

# Custom data YAML 생성
custom_data_yaml = {
    'train': '/content/drive/MyDrive/yolo_gray_dataset/train/images',
    'val': '/content/drive/MyDrive/yolo_gray_dataset/valid/images',
    'nc': 1,
    'names': ['sign'],
    'img_size': 64,
    'channels': 1
}

yaml_path = '/content/yolov5/data/custom_data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(custom_data_yaml, f, default_flow_style=False)
print(f"YAML file created at: {yaml_path}")

# Modified YOLOv5n model configuration
model_yaml = """
# YOLOv5n modified for 64x64 grayscale
nc: 1 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.25 # layer channel multiple
anchors:
 - [4,5, 8,10, 13,16] # P3/8
 - [23,29, 43,55, 73,105] # P4/16
backbone:
 [[-1, 1, Conv, [32, 3, 2, 1]], # 0-P1/2
 [-1, 1, Conv, [64, 3, 2]], # 1-P2/4
 [-1, 3, C3, [64]],
 [-1, 1, Conv, [128, 3, 2]], # 3-P3/8
 [-1, 3, C3, [128]],
 [-1, 1, Conv, [256, 3, 2]], # 5-P4/16
 [-1, 3, C3, [256]],
 [-1, 1, SPPF, [256, 5]], # 7
 ]
head:
 [[-1, 1, Conv, [128, 1, 1]],
 [-1, 1, nn.Upsample, [None, 2, 'nearest']],
 [[-1, 4], 1, Concat, [1]], # cat backbone P3
 [-1, 3, C3, [128, False]], # 11
 [-1, 1, Conv, [128, 3, 2]],
 [[-1, 6], 1, Concat, [1]], # cat head P4
 [-1, 3, C3, [256, False]], # 14 (P4/16)
 [[11, 14], 1, Detect, [nc, anchors]], # Detect(P3, P4)
 ]
"""

with open('/content/yolov5/models/yolov5n-gray.yaml', 'w') as f:
    f.write(model_yaml)
print("Modified YOLOv5n model configuration for 64x64 grayscale input saved.")

# Training command
!python train.py --img 64 --batch 64 --epochs 300 --data /content/yolov5/data/custom_data.yaml --cfg /content/yolov5/models/yolov5n-gray.yaml --weights '' --name yolov5n_gray_sign --cache
```

## Notes

- Ensure that your Google Drive paths are correct and accessible.
- Adjust hyperparameters like batch size and number of epochs as needed.
- Monitor the training process and use early stopping if the model starts to overfit.
- This setup is specifically for 64x64 grayscale images. Adjust the image size and channels if your dataset differs.
- The model configuration is a modified version of YOLOv5n. You might need to experiment with the architecture for optimal performance.

For more details on YOLOv5, refer to the [official YOLOv5 repository](https://github.com/ultralytics/yolov5).
