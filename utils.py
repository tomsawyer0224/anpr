import os
from ultralytics import YOLO, RTDETR

# import easyocr
import cv2
import torchshow as ts

# import numpy as np
import reader


def put_text(
    img,
    text,
    org,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1,
    color=(255, 255, 255),
    thickness=2,
    cvtBGR2RGB=False,
):
    """
    put highlighted text to image
    args:
        img: image (in BGR format)
        text: Text string to be drawn
        org: Bottom-left corner of the text string in the image
        fontFace: font type
        fontScale: Font scale factor that is multiplied by the font-specific base size
        color: Text color
        thickness: Thickness of the lines used to draw a text
        cvtBGR2RGB: convert image from BGR to RGB or not
    returns:
        img
    """
    (w, h), _ = cv2.getTextSize(
        text=text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )
    xmin, ymax = org
    ymin = ymax - h
    xmax = xmin + w
    img = cv2.rectangle(
        img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 0, 255), thickness=-1
    )
    img = cv2.putText(
        img=img,
        text=text,
        org=org,
        fontFace=fontFace,
        fontScale=1,
        thickness=2,
        color=(255, 255, 255),
    )
    if cvtBGR2RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return img


def get_video_properties(video):
    """
    get video properties
    args:
        video: file name or VideoCapture instance
    returns:
        video properties (in dict format)
    """
    if isinstance(video, str):
        video = cv2.VideoCapture(video)
    if not video.isOpened():
        return {}
    properties = {}
    properties["height"] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties["width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["fps"] = int(video.get(cv2.CAP_PROP_FPS))
    properties["n_frames"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = video.get(cv2.CAP_PROP_FOURCC)
    fourcc = int(fourcc)
    properties["fourcc"] = hex(fourcc)
    codec = (
        chr(fourcc & 0xFF)
        + chr((fourcc >> 8) & 0xFF)
        + chr((fourcc >> 16) & 0xFF)
        + chr((fourcc >> 24) & 0xFF)
    )
    properties["codec"] = codec
    return properties


def get_image_properties(image):
    """
    get image properties
    args:
        image: file name or image reached from cv2.imread()
    returns:
        image properties (in dict format)
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    properties = {}
    shape = image.shape
    properties["height"] = shape[0]
    properties["width"] = shape[1]
    properties["shape"] = shape
    properties["dtype"] = image.dtype
    return properties


def get_detector(detector):
    """
    args:
        detector: str, ultralytics instances
    returns:
        instance (yolov8 or rtdetr)
    """
    # try:
    if isinstance(detector, str):
        # supported detector
        d = ["yolov8", "rtdetr"]
        assert detector in d, f"detector should be in {d}"
        if d == "yolov8":
            detector = YOLO("./checkpoints/plate_dectection_yolov8m.pt")
        else:  # d == 'rtdetr'
            detector = RTDETR("./checkpoints/plate_dectection_rtdetrl.pt")
    else:
        # supported detector
        d = ["YOLO", "RTDETR"]
        assert (
            detector.__class__.__name__ in d
        ), f"detector should be an instance of {d}"
    return detector
    # except:
    # print('detector = "yolov8" or "rtdetr" or an instance of YOLO, RTDETR')


def get_reader(ocr_reader):
    """
    args;
        ocr_reader: str
    returns:
        class Reader's instance
    """
    if isinstance(ocr_reader, str):
        # supported readers
        r = ["easyocr", "paddleocr"]
        assert ocr_reader in r, f"ocr_reader should be in {r}"
        if ocr_reader == "easyocr":
            ocr_reader = reader.EasyocrReader()
        if ocr_reader == "paddleocr":
            ocr_reader = reader.PaddleocrReader()
    else:
        # supported readers
        r = ["EasyocrReader", "PaddleocrReader"]
        assert (
            ocr_reader.__class__.__name__ in r
        ), f"ocr_reader should be an instance of {r}"
    return ocr_reader


def get_files(directory):
    dirs = os.listdir(directory)
    files = []
    for d in dirs:
        f = os.path.join(directory, d)
        if os.path.isfile(f):
            files.append(f)
    return files


if __name__ == "__main__":
    directory = "./plate_dataset/test/images"
    files = get_files(directory)
    print(f"number of files in {directory}: {len(files)}")
    print(f"some files are {files[:10]}")
