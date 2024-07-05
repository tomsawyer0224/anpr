This is a personal project, for educational purpose only!

About this project:
  1. ANPR - Automatic Number Plate Recognition, includes 2 stages:
     First stage (Plate Detection): the Detector (YOLOv8 or RT-DETR) detects plates
     Second stage (Number Reading): the Reader (easyocr or PaddleOCR) reads text on the plates
  2. Supports 2 detectors and 2 readers >> 4 combinations. Their recognition time are the same (~0.11s/plate), see "benchmark_results" folder for more details.
  3. Pre-trained models (detectors) are stored in the file "./checkpoints/ckpts.txt". They have been trained on the plate dataset (https://universe.roboflow.com/anpr-bccrx/anpr-bpzor) by the Ultralytics library.

How to use:
  1. Clone this repo, cd to anpr
  2. Install the requirements: pip install -q -r requirements.txt
  3. Download pre-trained models from file ./checkpoints/ckpts.txt and store them to the "checkpoints" folder.
  4. Example: \
     from anpr import ANPR \
     plate_recognizer = ANPR( \
          detector = 'rtdetr', \
          reader = 'paddleocr' \
      ) \
     image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"] \
     images = plate_recognizer.read_from_image(source = image_path)
    
