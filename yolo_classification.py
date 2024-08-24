import cv2
import numpy as np
import os
import tensorflow as tf

# YOLO 모델 파일 및 구성 파일 로드
net = cv2.dnn.readNet("/home/hyunsoo/hyunsoo/yolov3.weights", "/home/hyunsoo/hyunsoo/yolov3.cfg")
layer_names = net.getLayerNames()

# getUnconnectedOutLayers()가 스칼라 값을 반환하는 경우 처리
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_layers - 1]]

# YOLO의 클래스 라벨을 "traffic sign"으로만 제한
classes = ["traffic sign"]

# 테스트할 이미지 경로 설정
image_path = "/home/hyunsoo/hyunsoo/30_1.jpg"
image = cv2.imread(image_path)

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print(f"Error: Unable to open image at {image_path}")
    exit()

height, width, channels = image.shape

# YOLO 입력 전처리
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# YOLO 추론 수행
outs = net.forward(output_layers)

# 검출된 바운딩 박스 및 클래스 저장
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만
            # 바운딩 박스 좌표 및 크기 계산
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 좌상단 좌표 계산
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Maximum Suppression 적용
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 저장된 모델 불러오기
model = tf.keras.models.load_model('speed_sign_cnn_model.h5')

# 클래스 라벨 설정
class_labels = ['30', '40', '50', '60', '70', '80', '90']

# "traffic sign"만 크롭하여 분류 모델에 입력
for i, index in enumerate(indexes):
    x, y, w, h = boxes[index]

    # 교통 표지판 이미지 크롭
    cropped_img = image[y:y+h, x:x+w]

    # 그레이스케일 변환 및 128x128 크기로 리사이즈
    gray_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_cropped_img, (128, 128))
    normalized_img = resized_img.astype('float32') / 255.0
    reshaped_img = normalized_img.reshape(1, 128, 128, 1)

    # CNN 모델을 사용한 예측 수행
    prediction = model.predict(reshaped_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    # 예측 결과 출력
    print(f"Detected traffic sign {i}: Predicted speed limit is: {predicted_class_label} km/h")

    # 크롭된 이미지 저장 (옵션)
    output_file_path = f"cropped_{i}.jpg"
    cv2.imwrite(output_file_path, resized_img)
    print(f"Cropped and grayscaled image saved as: {output_file_path}")

print("Detection, cropping, and classification completed.")

