
import numpy as np
import os
import random
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
    
class PersonReidentification(Model):

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
        result =  super().infer(in_frame)
        return np.delete(result, 1)


class Tracker:
    def __init__(self):
        # 識別情報のDB
        self.identifysDb = None
        # 中心位置のDB
        self.center = []
    
    def __getCenter(self, person):
        x = person[0] - person[2]
        y = person[1] - person[3]
        return (x,y)

    def __getDistance(self, person, index):
        (x1, y1) = self.center[index]
        (x2, y2) = self.__getCenter(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    def __isOverlap(self, persons, index):
        [x1, y1, x2, y2] = persons[index]
        for i, person in enumerate(persons):
            if(index == i):
                continue
            if(max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    def getIds(self, identifys, persons):
        if(identifys.size==0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in persons:
                self.center.append(self.__getCenter(person))
        
        #print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            personId = ids[i]
            d = self.__getDistance(persons[i], personId)
            #print("personId:{} {} distance:{}".format(personId,similarity[personId], d))
            # 0.9以上で、重なりの無い場合、識別情報を更新する
            if(similarity[personId] > 0.9):
                if(self.__isOverlap(persons, i) == False):
                    self.identifysDb[personId] = identifys[i]
            # 0.4以下で、距離が離れている場合、新規に登録する
            elif(similarity[personId] < 0.4):
                if(d > 800):
                    #print("distance:{} similarity:{}".format(d, similarity[personId]))
                    self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                    self.center.append(self.__getCenter(persons[i]))
                    ids[i] = len(self.identifysDb) - 1
                    #print("> append DB size:{}".format(len(self.identifysDb)))

        #print(ids)
        # 重複がある場合は、信頼度の低い方を無効化する
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if(e == i):
                    continue
                if(a == b):
                    if(similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        #print(ids)
        return ids

    # コサイン類似度
    # https://github.com/kodamap/person_reidentification
    def __cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (
            np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
        )

def get_person(detection):
    return [int(v) for v in detection[:4]]

device = "CPU"
cpu_extension = None
ie_core = IECore()
if device == "CPU" and cpu_extension:
    ie_core.add_extension(cpu_extension, "CPU")

THRESHOLD= 0.65
person_detector = PersonDetector("model/person-detection-retail-0013", device, ie_core, THRESHOLD, num_requests=2)
person_reidentification = PersonReidentification("model/person-reidentification-retail-0265", device, ie_core, THRESHOLD, num_requests=2)
tracker = Tracker()


SCALE = 0.75

movies = [
  'pexels-richarles-moral-1338598-1920x1080-30fps.mp4',
  'pexels-pixabay-855565-1920x1080-24fps.mp4',
  'istockphoto-1141803334-640_adpp_is.mp4'
]
cap = cv2.VideoCapture(f'test/movie/{movies[0]}')
# Webカメラ
# cap = cv2.VideoCapture(0)
TRACKING_MAX=50
colors = []
for i in range(TRACKING_MAX):
    colors.append((
        random.randint(80, 255),
        random.randint(80, 255),
        random.randint(80, 255)
    ))
n = 0
while cap.isOpened():
    n += 1
    ret, frame = cap.read()
    
    if not ret:# ループ再生
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if(frame is None):
        continue

    if n % 4:
        continue

    detections = person_detector.infer(frame)
    persons = [person[:4] for person in detections]

    identifys = np.zeros((len(persons), 255))

    for i, person in enumerate(detections):
        x1, y1, x2, y2 = get_person(person)
        conf = person[4]
        
        img = frame[y1:y2, x1:x2]
        h, w = img.shape[:2]
        if h == 0 or w == 0: continue
        identifys[i] = person_reidentification.infer(img)
        
    ids = tracker.getIds(identifys, persons)
    
    for i, person in enumerate(detections):
        x1, y1, x2, y2 = get_person(person)
        if ids[i] == -1: continue
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
        frame = cv2.putText(frame, '[ID {}] {:.02f}%'.format(int(ids[i]), conf * 100), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, colors[i], 1, cv2.LINE_AA )

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, ((int(w * SCALE), int(h * SCALE))))

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()