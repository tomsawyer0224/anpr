# This is a personal project, for educational purposes only!
# About this project:
1. **ANPR** or **Automatic Number Plate Recognition** is a Recognition Model that includes two stages:
   First stage (Plate Detection): Detector (YOLOv8 or RT-DETR) detects plates
   Second stage (Number Reading): Reader (easyocr or PaddleOCR) reads text on the plates
2. Supports 2 detectors and 2 readers ⟶ 4 combinations. Their recognition times are the same (~0.11s/plate), see the "benchmark_results" folder for more details.
3. Pre-trained models (detectors) are stored in the file "./checkpoints/ckpts.txt". They have been trained on the [plate dataset](https://universe.roboflow.com/anpr-bccrx/anpr-bpzor) by the Ultralytics library.
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
images = plate_recognizer.read_from_image(source = image_path)https://universe.roboflow.com/anpr-bccrx/anpr-bpzor
```
5. Some results: \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/98f7c359-b211-4e8c-aeff-5a5da70df00e) \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/f15c5095-2308-4044-bd67-048e6c87b784) \
     ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/b75f8696-c2a4-470d-8c47-f13448178a3c)

