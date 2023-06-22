# インポート (適宜pipでインストールする)
import imutils
import numpy as np
import cv2
import datetime
import os
import glob


def load_features():
    fes = []
    files = glob.glob(os.path.join('faces', "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        fes.append((user_id, feature))
    return fes

FEATURES = load_features()

COSINE_THRESHOLD = 0.72

# VideoCaptureをオープン
cap = cv2.VideoCapture(0)

# モデルを読み込む
"""
https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view
"""
detect_net = cv2.dnn.readNetFromCaffe('model/face/deploy.prototxt', 'model/face/res10_300x300_ssd_iter_140000.caffemodel')
face_recognizer = cv2.FaceRecognizerSF.create('model/face/face_recognizer_fast.onnx', '')

def match(target):
    results = []
    for element in FEATURES:
        user_id, feature2 = element
        score = face_recognizer.match(target, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        results.append(score)
    if len(results):
        i = np.argmax(results)
        s = results[i]
        if s > COSINE_THRESHOLD:
            return True, (FEATURES[i][0], s)
    return False, ("", 0.0)

okay = 0

while True:
    ret, frame = cap.read()

    img = imutils.resize(frame, width=764)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    detect_net.setInput(blob)
    detections = detect_net.forward()

    copy = imutils.rotate(img, 0)

    outs = []
    #bgr

    okay_flag = False
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            color = (0, 0, 255)
            if confidence > 0.95:
                color = (80, 230, 90)
            elif confidence > 0.8:
                color = (80, 230, 190)

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_img = copy[startY:endY, startX:endX]

            # aligned_face_img = face_recognizer.alignCrop(face_img, copy)

            outs.append( face_img )

            feature = face_recognizer.feature(face_img)
            result, user = match(feature)
            
            id, score = user if result else ("?", 0.0)

            if score >= COSINE_THRESHOLD:
                okay_flag = True

            text = "{} ({:.2f}): {:.2f}%".format(id, score, confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 15
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, color if result else (0, 0, 255), 2)

    if okay_flag:
        okay += 1
    else:
        okay = 0

    if okay > 20:
        print('Okay!')
        okay = 0

    cv2.imshow("Face Detection", img)
    k = cv2.waitKey(1) & 0xff
    #if -1 < k: print(k)
    if k == ord('s'):
        if not os.path.exists('faces'):
            os.mkdir('faces')
        cpath = lambda n, ext: f"faces/{str(n).zfill(4)}{ext}"
        i = 0
        for out in outs:
            while True:
                i += 1
                if not os.path.exists( cpath(i, '.npy') ):
                    f = face_recognizer.feature(out)
                    n = cpath(i, '')
                    np.save(n, f)
                    print(f'> {n}.npy')
                    print(f)
                    print('=' * 10)
                    break
        FEATURES = load_features()
    elif k == ord('q') or k == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
