# ANPR - Automatic Number Plate Recognition
ANPR is a Recognition Model that includes two stages:
- First stage: Plate Detection (detects all plates in the images).
- Second stage: Number Reading (reads the text on the detected plates).
# About this project
- This is a personal project, for educational purposes only!
- This project was built to evaluate the performance when combining two detectors (YOLOv8, RT-DETR) and two readers (easyocr, PaddleOCR).
# Experiment
1. Training
- The detectors:
    - Model: yolov8m, yolov8n, rt-detrl.
    - Dataset: [plate dataset](https://universe.roboflow.com/anpr-bccrx/anpr-bpzor).
    - Platform: Kaggle free (T4 GPU).
- The readers are used directly.
984 image on test
4. Pre-trained models (detectors) are stored in the file "./checkpoints/ckpts.txt". They have been trained on the [plate dataset](https://universe.roboflow.com/anpr-bccrx/anpr-bpzor) by the Ultralytics library.
# How to use:
1. Clone this repo, cd to anpr.
2. Install the requirements: pip install -q -r requirements.txt.
3. Download the pre-trained models (in ./checkpoints/ckpts.txt) and store them in the "checkpoints" folder.
4. Example:
```
from anpr import ANPR
plate_recognizer = ANPR(
    detector = 'rtdetr',
    reader = 'paddleocr'
)
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
images = plate_recognizer.read_from_image(source = image_path)
```
5. Some results: \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/98f7c359-b211-4e8c-aeff-5a5da70df00e) \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/f15c5095-2308-4044-bd67-048e6c87b784) \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/b75f8696-c2a4-470d-8c47-f13448178a3c)

