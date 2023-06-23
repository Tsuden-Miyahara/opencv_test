
import imutils
import numpy as np
import cv2
import hashlib
import datetime
import os
import sys
import glob
import copy

from _yunet import YuNetONNX, draw_debug

def load_features():
    fes = []
    files = glob.glob(os.path.join('faces', "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        fes.append((user_id, feature))
    return fes


def get_hash_pass():
    txt = datetime.utcnow().strftime('%Y-%m-%d')
    return hashlib.sha256(txt.encode()).hexdigest()

FEATURES = load_features()

COSINE_THRESHOLD = 0.72

QR_PASSWORD = 'tsuden_guest'

# VideoCaptureをオープン
cap = cv2.VideoCapture(0)

# モデルを読み込む

face_detector = cv2.FaceDetectorYN.create('model/face/face_detection_yunet_120x160.onnx', '', (160, 120))
yunet = YuNetONNX(
    # model_path='model/face/yunet_2023mar.onnx'
    model_path='model/face/face_detection_yunet_120x160.onnx'
)
face_recognizer = cv2.FaceRecognizerSF.create('model/face/face_recognizer_fast.onnx', '')

qcd_detector = cv2.QRCodeDetector()

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
    okay_flag = False
    outs = []
    ret, img = cap.read()

    # img = imutils.resize(frame, width=764)
    clone = copy.deepcopy(img)
    w, h = [160, 120]

    bboxes, landmarks, scores = yunet.inference(img)

    image_width, image_height = img.shape[1], img.shape[0]
    debug_image = copy.deepcopy(img)


    """
    face_detector.setInputSize(img.shape[:2][::-1])
    _, faces = face_detector.detect(img)
    if len(fs):
        af = face_recognizer.alignCrop(img, fs[0])
        cv2.imwrite('test.jpg', af)
    """

    for bbox, landmark, score in zip(bboxes, landmarks, scores):
        if COSINE_THRESHOLD > score:
            continue
        
        # 顔バウンディングボックス
        x1 = int(image_width  * (bbox[0] / w))
        y1 = int(image_height * (bbox[1] / h))
        x2 = int(image_width  * (bbox[2] / w)) + x1
        y2 = int(image_height * (bbox[3] / h)) + y1



        """
        aligned_face = face_recognizer.alignCrop(img, bbox)
        face_feature = face_recognizer.feature(aligned_face)
        outs.append( aligned_face )
        """
        face = debug_image[y1:y2, x1:x2]
        face_feature = face_recognizer.feature(face)
        outs.append( face )


        result, user = match(face_feature)
        
        id, score = user if result else ("?", 0.0)
        color = (80, 230, 90) if result else (0, 0, 255)

        if score > COSINE_THRESHOLD:
            okay_flag = True


        text = "{} ({:.2f})".format(id, score)


        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

        # 顔キーポイント
        for _, landmark_point in enumerate(landmark):
            x = int(image_width * (landmark_point[0] / w))
            y = int(image_height * (landmark_point[1] / h))
            cv2.circle(img, (x, y), 2, color, 2)



    isQ, q_decoded, q_pos, _ = qcd_detector.detectAndDecodeMulti(img)
    if isQ:
        for s, p in zip(q_decoded, q_pos):
            if s == QR_PASSWORD:
                color = (80, 230, 90)
                txt = 'Valid'
                okay_flag = True
            else:
                color = (0, 0, 255)
                txt = 'Invalid'
            cv2.polylines(img, [p.astype(int)], True, color, 2)
            cv2.putText(img, txt, p[0].astype(int),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



    if okay_flag:
        okay += 1
    else:
        okay = 0

    if okay > 5:
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
                    n = cpath(i, '')
                    ft = face_recognizer.feature(out)
                    np.save(n, ft)
                    print(f'> {n}.npy')
                    print(ft)
                    print('=' * 10)

                    cv2.imwrite(cpath(i, '.jpg'), out)
                    break
        FEATURES = load_features()
    elif k == ord('q') or k == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
