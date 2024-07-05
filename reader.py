import torch
import cv2
import easyocr
from paddleocr import PaddleOCR 

class BaseReader:
    def read_plate(self, plate):
        '''
        read text from plate
        args:
            plate: numpy array
        returns:
            text is read from plate
        '''
        pass
        
class EasyocrReader(BaseReader):
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu = torch.cuda.is_available())
    def read_plate(self, plate):
        if len(plate.shape) == 3:
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plater = cv2.blur(plate, (5,5))
        _, plate = cv2.threshold(plate,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        texts = self.reader.readtext(plate)
        if texts:
            plate_number = ''
            for text in texts:
                plate_number += text[1]
            return plate_number
        else:
            return 'UNK'

class PaddleocrReader(BaseReader):
    def __init__(self):
        self.reader = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            use_gpu=torch.cuda.is_available(),
            show_log = False
        )
    def read_plate(self, plate):
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        # texts in the images
        texts_in_images = self.reader.ocr(img = plate)
        # only one 1 image -> only one texts
        texts = texts_in_images[0]
        if texts:
            #print('@@@-texts-@@@', texts)
            plate_number = ''
            for text in texts:
                #print(f'---text---', text)
                plate_number += text[1][0]
            return plate_number
        else:
            return 'UNK'



