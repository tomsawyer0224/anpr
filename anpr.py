import torch
import numpy as np
import time

# import os
import pandas as pd
import cv2
import easyocr
from ultralytics import YOLO
import torchshow as ts
import utils


class ANPR:
    def __init__(
        self,
        detector,
        reader,
    ):
        """
        args:
            detector: plate recognizer
            reader: ocr reader
        """
        self.gpu = torch.cuda.is_available()  # True or False
        self.detector = utils.get_detector(detector)
        self.reader = utils.get_reader(reader)

    def read_from_image(self, source):
        """
        args:
            source: image paths, numpy arrays
        returns:
            list of image (in RGB format)
        """

        results = self.detector.predict(
            source=source,
            # persist = True,
            device=0 if self.gpu else None,
            verbose=False,
        )

        images = []
        for result in results:
            image = result.orig_img
            boxes = result.boxes

            if boxes:
                boxes_xyxy = boxes.xyxy
                boxes_xyxy = torch.where(boxes_xyxy >= 0, boxes_xyxy, 0)
                for box_id, box in enumerate(boxes_xyxy):
                    box_id = box_id + 1
                    xmin, ymin, xmax, ymax = box.int().tolist()
                    plate = image[ymin : ymax + 1, xmin : xmax + 1, :]
                    plate_number = self.reader.read_plate(plate)
                    image = cv2.rectangle(
                        img=image,
                        pt1=(xmin, ymin),
                        pt2=(xmax, ymax),
                        color=(0, 0, 255),
                        thickness=2,
                    )
                    image = utils.put_text(
                        img=image,
                        text=f"id:{box_id},{plate_number}",
                        org=(xmin, ymin),
                    )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        # ts.show(images)
        return images

    def benchmark(self, image_list):
        """
        args:
            source: list of image paths
        returns:
            time
        """
        if not isinstance(image_list, list):
            image_list = list(image_list)
        image_ids = []
        detect_times = []
        read_times = []
        n_boxes_per_img = []
        for image_id, image_path in enumerate(image_list):
            # print(f'processing {image_path}')
            # detecting time
            dst = time.time()
            results = self.detector.predict(
                source=image_path,
                # persist = True,
                device=0 if self.gpu else None,
                verbose=False,
            )
            det = time.time()
            # print(f'results of detection phase, n_results = {len(results)}')
            # print(results)
            # print('--'*20)
            # return 1
            # images = []
            # reading time
            rst = time.time()
            # actually we have only one result, because we passed 1 image to predict method
            for result in results:
                image = result.orig_img
                boxes = result.boxes
                # print(f'type(boxes): {type(boxes)}')
                # print(f'type(boxes.id): {type(boxes.id)}', boxes.id)
                # print(f'type(boxes.xyxy): {type(boxes.xyxy)}', boxes.xyxy)
                # print('---boxes in single result')
                # print('---boxes\n', boxes)
                if boxes:
                    n_boxes_per_img.append(len(boxes))
                    # print(boxes)
                    # return 1
                    boxes_xyxy = boxes.xyxy
                    boxes_xyxy = torch.where(boxes_xyxy >= 0, boxes_xyxy, 0)
                    for box_id, box in enumerate(boxes_xyxy):
                        # box_id = box_id.int().item()
                        box_id += 1
                        xmin, ymin, xmax, ymax = box.int().tolist()
                        plate = image[ymin : ymax + 1, xmin : xmax + 1, :]
                        plate_number = self.reader.read_plate(plate)
                        image = cv2.rectangle(
                            img=image,
                            pt1=(xmin, ymin),
                            pt2=(xmax, ymax),
                            color=(0, 0, 255),
                            thickness=2,
                        )
                        image = utils.put_text(
                            img=image,
                            text=f"id:{box_id},{plate_number}",
                            org=(xmin, ymin),
                        )
                else:
                    n_boxes_per_img.append(0)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # images.append(image)
            # print(f'done {image_path}')
            ret = time.time()
            image_ids.append(image_id)
            detect_times.append(det - dst)
            read_times.append(ret - rst)
        results = pd.DataFrame(
            {
                "image": image_ids,
                "detect_time": detect_times,
                "read_time": read_times,
                "n_boxes": n_boxes_per_img,
            }
        )
        return results

    def read_from_video(self, source):
        pass
