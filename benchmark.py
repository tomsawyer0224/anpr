import pandas as pd
import os
import cv2

# import torch
from anpr import ANPR
import utils


class Benchmark:
    def __init__(self):
        self.anprs = {
            "yolov8_easyocr": ANPR(detector="yolov8", reader="easyocr"),
            "yolov8_paddleocr": ANPR(detector="yolov8", reader="paddleocr"),
            "rtdetr_easyocr": ANPR(detector="rtdetr", reader="easyocr"),
            "rtdetr_paddleocr": ANPR(detector="rtdetr", reader="paddleocr"),
        }

    def benchmark(self, image_list, save_results=None):
        benchmark_results = {}
        gpus = {}

        avg_detect_times = []
        avg_read_times = []
        avg_boxes = []
        anpr_types = []
        use_gpus = []
        for anpr_type, anpr in self.anprs.items():
            gpus[anpr_type] = anpr.gpu
            result = anpr.benchmark(image_list)
            benchmark_results[anpr_type] = result

            avg = result.mean()
            avg_detect_times.append(avg["detect_time"])
            avg_read_times.append(avg["read_time"])
            avg_boxes.append(avg["n_boxes"])
            anpr_types.append(anpr_type)
            use_gpus.append(anpr.gpu)
        avg = pd.DataFrame(
            {
                "anpr_type": anpr_types,
                "avg_detect_time": avg_detect_times,
                "avg_read_time": avg_read_times,
                "avg_box": avg_boxes,
                "use_gpu": use_gpus,
            }
        )
        if save_results:
            result_path = "./benchmark_results"
            os.makedirs(result_path, exist_ok=True)
            for anpr_type, result in benchmark_results.items():
                chip_type = "GPU" if gpus[anpr_type] else "CPU"
                filename = anpr_type + "_" + chip_type + ".csv"
                save_dir = os.path.join(result_path, filename)
                result.to_csv(save_dir, index=False)
            chip_type = "GPU" if gpus["yolov8_easyocr"] else "CPU"
            filename = "average" + "_" + chip_type + ".csv"
            avg.to_csv(os.path.join(result_path, filename))
        print(avg)


if __name__ == "__main__":
    bm = Benchmark()
    image_list = utils.get_files("./plate_dataset/test/images")
    bm.benchmark(image_list, save_results=True)
