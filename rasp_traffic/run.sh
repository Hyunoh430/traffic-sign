IMAGE_PATH='/home/Pi/Documents/traffic/captured_image.png'
RESULT_PATH='/home/Pi/Documents/traffic/result_image.png'

libcamera-still -o "$IMAGE_PATH" --width 128 --height 128

python3 /home/Pi/Documents/traffic/gray_run.py "$IMAGE_PATH" "$RESULT_PATH"


