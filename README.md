This is a personal project, for educational purpose only!
About this project:
  1. ANPR - Automatic Number Plate Recognition, includes 2 stages:
     First stage (Plate Detection): the Detector (YOLOv8 or RT-DETR) detects plates
     Second stage (Number Reading): the Reader (easyocr or PaddleOCR) reads text on the plates
  2. Supports 2 detectors and 2 readers >> 4 combinations. Their recognition time are the same, see "benchmark_results" folder.
