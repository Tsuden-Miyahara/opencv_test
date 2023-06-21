
import numpy as np
import os
import cv2
from openvino.inference_engine import IECore
from model import Model

class PersonDetector(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):
        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result = super().infer(in_frame)

        detections = []
        height, width = frame.shape[:2]
        for r in result[0][0]:
            conf = r[2]
            if(conf > self.__threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])
        return detections
    


device = "CPU"
cpu_extension = None
ie_core = IECore()
if device == "CPU" and cpu_extension:
    ie_core.add_extension(cpu_extension, "CPU")

THRESHOLD= 0.8
person_detector = PersonDetector("model/person-detection-retail-0013", device, ie_core, THRESHOLD, num_requests=2)

SCALE = 1.0

input_dir = os.path.join('test', 'input')
output_dir = os.path.join('test', 'output')

for p in os.listdir(input_dir):
    name, ext = os.path.splitext(p)
    if not ext.lower() in ('.jpg', '.jpeg', '.png'):
        continue

    img_from = os.path.join(input_dir, p)
    img_to = os.path.join(output_dir, f'{name}.png')
    print('=' * 8)
    print(f'* image: {p}')

    frame = cv2.imread(img_from)
    detections = person_detector.infer(frame)


    for person in detections:
        x1 = int(person[0])
        y1 = int(person[1])
        x2 = int(person[2])
        y2 = int(person[3])
        conf = person[4]
        print("{:.03f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
        color = (0, 255, 0)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(2 * SCALE))
        frame = cv2.putText(frame, '{:.03f}'.format(conf), (x1, y1), cv2.FONT_HERSHEY_PLAIN, int(1 * SCALE), color, int(1 * SCALE), cv2.LINE_AA )

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, ((int(w * SCALE), int(h * SCALE))))

    cv2.imwrite(img_to, frame)


