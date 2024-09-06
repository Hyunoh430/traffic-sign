# YOLOv5 Grayscale 64x64 Object Detection

This project modifies the YOLOv5 model to work with 64x64 grayscale images, creating a smaller and faster version of the original model.

## Project Overview

We've adapted the YOLOv5n model to:
1. Accept 64x64 grayscale images as input
2. Reduce the model size and complexity
3. Focus on a single class detection task

## Setup Instructions

1. Clone the YOLOv5 repository:
   ```
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Modify the YOLOv5 model configuration:
   Edit the file `models/yolov5n.yaml` with the following changes:

   ```yaml
   # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
   # Parameters
   nc: 1  # number of classes
   depth_multiple: 0.25  # model depth multiple
   width_multiple: 0.20  # layer channel multiple
   anchors:
     - [10,13, 16,30, 33,23]  # P3/8
     - [30,61, 62,45, 59,119]  # P4/16
     - [116,90, 156,198, 373,326]  # P5/32

   # YOLOv5 v6.0 backbone
   backbone:
     # [from, number, module, args]
     [[-1, 1, Conv, [64, 1, 2]],  # 0-P1/2
      # ... (rest of the backbone remains the same)
     ]

   # ... (head remains the same)
   ```

   Key changes:
   - Set `nc: 1` for single class detection
   - Reduced `depth_multiple` and `width_multiple`
   - Changed the first Conv layer to accept 1 channel input: `[64, 1, 2]`

4. Modify the data loading process:
   Edit the file `utils/dataloaders.py` and update the `LoadImages` class's `__next__` method as follows:

   ```python
   class LoadImages:
       # ... (previous code remains the same)

       def __next__(self):
           """Advances to the next file in the dataset, raising StopIteration if at the end."""
           if self.count == self.nf:
               raise StopIteration
           path = self.files[self.count]

           if self.video_flag[self.count]:
               # Read video
               self.mode = "video"
               for _ in range(self.vid_stride):
                   self.cap.grab()
               ret_val, im0 = self.cap.retrieve()
               while not ret_val:
                   self.count += 1
                   self.cap.release()
                   if self.count == self.nf:  # last video
                       raise StopIteration
                   path = self.files[self.count]
                   self._new_video(path)
                   ret_val, im0 = self.cap.read()

               self.frame += 1
               # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
               s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

           else:
               # 이미지 처리 로직 수정
               self.count += 1
               im0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 읽기
               assert im0 is not None, f"Image Not Found {path}"
               s = f"image {self.count}/{self.nf} {path}: "

           if self.transforms:
               im = self.transforms(im0)  # transforms
           else:
               # 이미지 크기 조정 및 처리
               im = cv2.resize(im0, (64, 64), interpolation=cv2.INTER_LINEAR)
               im = np.expand_dims(im, axis=-1)  # (64, 64, 1)로 차원 추가
               im = im.transpose((2, 0, 1))  # CHW 형식으로 변환
               im = np.ascontiguousarray(im)

           return path, im, im0, self.cap, s

       # ... (rest of the code remains the same)
   ```

5. Prepare your dataset:
   Create a `custom_data.yaml` file in the `data` directory:

   ```yaml
   path: /path/to/your/dataset  # dataset root dir
   train: images/train  # train images (relative to 'path')
   val: images/val  # val images (relative to 'path')

   nc: 1  # number of classes
   names: ['your_class_name']  # class names
   ```

## Training the Model

To train the model, run the following command:

```
python train.py --img 64 --batch 64 --epochs 100 --data custom_data.yaml --cfg models/yolov5n.yaml --weights '' --name tiny_gray_yolo
```

Note: We use `--weights ''` to train from scratch since we modified the input channels.

## Inference

After training, you can run inference on new images using:

```
python detect.py --source path/to/your/images --weights runs/train/tiny_gray_yolo/weights/best.pt --img 64
```

## Key Changes Summary

1. YOLOv5n model configuration (`models/yolov5n.yaml`):
   - Reduced model complexity
   - Changed input to accept grayscale images

2. Data loading (`utils/dataloaders.py`):
   - Modified to load grayscale images while maintaining video handling capability
   - Resized images to 64x64 when transforms are not provided
   - Adjusted image processing for grayscale input, including adding a channel dimension

These modifications allow the YOLOv5 model to work efficiently with 64x64 grayscale images while still supporting video input.

## Notes

- Ensure your dataset consists of grayscale images for optimal performance.
- The model's performance may vary depending on the complexity of your detection task.
- You might need to adjust anchor sizes in the YAML file if your objects differ significantly from the default COCO dataset.
- The data loading process now handles both image and video inputs, with images being processed as 64x64 grayscale.

## Troubleshooting

If you encounter issues:
1. Verify all file modifications are correct.
2. Ensure your dataset is properly formatted and paths are correct.
3. Check that input images are in grayscale format.
4. If using video inputs, ensure the video handling part of the code works as expected.

For persistent issues, refer to the YOLOv5 GitHub repository or seek help in the project's issues section.


## Saved Files Location

After training, the model weights and other related files will be saved in the following directory structure:

```
runs/
└── train/
    └── tiny_gray_yolo/  # This name comes from the --name argument in the training command
        ├── weights/
        │   ├── best.pt      # Best model weights
        │   └── last.pt      # Latest model weights
        ├── confusion_matrix.png
        ├── results.csv
        ├── results.png
        └── ... (other training artifacts)
```

Key files:
- `best.pt`: The weights of the model that performed best on the validation set.
- `last.pt`: The weights of the model at the end of training.
- `results.csv`: A CSV file containing training metrics for each epoch.
- `results.png`: A plot of the training metrics.

You can find these files in the `runs/train/tiny_gray_yolo/` directory (or whatever name you specified with the `--name` argument during training).

To use the trained model for inference, you would typically use the `best.pt` weights file. For example:

```
python detect.py --source path/to/your/images --weights runs/train/tiny_gray_yolo/weights/best.pt --img 64
```

Remember to adjust the path if you used a different name during training.