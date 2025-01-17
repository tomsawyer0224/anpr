# ANPR: Automatic Number Plate Recognition
ANPR is a Recognition Model that includes two stages:
- First stage: Plate Detection (detects all plates in the images).
- Second stage: Number Reading (reads the text on the detected plates).
# About this project
- This is a personal project, for educational purposes only!
- This project was built to evaluate the performance when combining two detectors (YOLOv8, RT-DETR) and two readers (easyocr, PaddleOCR).
# Experiment
1. **Training**
    - The detectors:
        - Model: yolov8m, yolov8n, rt-detrl.
        - Dataset: [plate dataset](https://universe.roboflow.com/anpr-bccrx/anpr-bpzor).
        - Library: Ultralytics.
        - Platform: Kaggle Free (with a T4 GPU).
    - The readers are used directly.
2. **Evaluate the performance of these ANPR models**
    - Models: yolov8_easyocr, yolov8_paddleocr, rtdetr_easyocr, and rtdetr_paddleocr.
    - Test dataset: 984 images.
    - Platform: Google Colab Free (with a T4 GPU).
3. **Results**
    - All models take the same time (0.11-0.12s) to recognize a plate.
        ```
        anpr_type	    avg_detect_time	  avg_read_time	         avg_box	        use_gpu
        yolov8_easyocr	    0.08119338698195289	  0.034601540981962824	 1.083415112855741	True
        yolov8_paddleocr    0.08777752872538169	  0.032225956743668994	 1.083415112855741	True
        rtdetr_easyocr	    0.08422949433443708	  0.03484810931174865	 1.083415112855741	True
        rtdetr_paddleocr    0.0851054154153661	  0.030265903098543444	 1.083415112855741	True
        ```
    - These models can recognize the plates in the images that are not too blurry. \
    ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/98f7c359-b211-4e8c-aeff-5a5da70df00e) \
    ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/f15c5095-2308-4044-bd67-048e6c87b784) \
    ![image](https://github.com/tomsawyer0224/anpr/assets/130035084/b75f8696-c2a4-470d-8c47-f13448178a3c)
3. **Conclusions**
    - The performance is acceptable when using small models.
    - The recognition time can be reduced by using a better GPU.
# How to use
1. Clone this repo and cd into anpr.
2. Install the requirements: pip install -q -r requirements.txt.
3. Download the pre-trained models (in ./checkpoints/ckpts.txt) and store them in the "checkpoints" folder.
4. Example
```
from anpr import ANPR
plate_recognizer = ANPR(
    detector = 'rtdetr',
    reader = 'paddleocr'
)
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
images = plate_recognizer.read_from_image(source = image_path)
```

