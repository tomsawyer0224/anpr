from ultralytics import YOLO, RTDETR
from PIL import Image
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import easyocr
from anpr import ANPR
import reader
import utils

#source = './plate_dataset/test/images/0039ac4bfb8bd69d_jpg.rf.8ed0bdac228b8da2b163f8f4ec3ee12d.jpg'
#pil_img = Image.open(source)
#plt.imshow(pil_img)
#plt.show()

detectors = {
    'yolov8':'yolov8',
    'rtdetr':'rtdetr',
    #'YOLO':YOLO('./checkpoints/plate_dectection_yolov8m.pt'),
    #'RTDETR':RTDETR('./checkpoints/plate_dectection_rtdetrl.pt')
}

#ocr_reader = 'easyocr'
#ocr_reader = 'paddleocr'
#ocr_reader = reader.EasyocrReader()
#ocr_reader = reader.PaddleocrReader()

ocr_readers = {
    'easyocr':'easyocr',
    'paddleocr':'paddleocr',
    #'EasyocrReader':reader.EasyocrReader(),
    #'PaddleocrReader':reader.PaddleocrReader()
}
'''
for dkey, dvalue in zip(detectors.keys(), detectors.values()):
    detector = dvalue
    for rkey, rvalue in zip(ocr_readers.keys(), ocr_readers.values()):
        ocr_reader = rvalue
        plate_recognizer = ANPR(detector = detector, reader = ocr_reader)
        print(f'---initialize plate recognizer from {dkey},{rkey} successfully!---')
'''
plate_recognizer = ANPR(
        detector = detectors['rtdetr'], 
        reader = ocr_readers['paddleocr']
    )
image_list = utils.get_files('./plate_dataset/test/images')
#print(image_list[:10])
images = plate_recognizer.read_from_image(source = image_list[:10])

