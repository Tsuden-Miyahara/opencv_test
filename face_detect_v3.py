
import imutils
import numpy as np
import cv2
import hashlib
import datetime
import os
import sys
import glob
import copy
import qrcode
from PIL import Image


def load_features():
    fes = []
    files = glob.glob(os.path.join('faces', "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        fes.append((user_id, feature))
    return fes


def get_hash_pass():
    txt = datetime.datetime.utcnow().strftime('%Y%m%d')
    return f'tsuden_{hashlib.sha256(txt.encode()).hexdigest()}_guest'


# variables start
FEATURES = load_features()

COSINE_THRESHOLD = 0.72

PASS_SCORE = 10

QR_PASSWORD = get_hash_pass()
QR_MASTER_PASSWORD = f'tsuden_{hashlib.sha256("19760312".encode()).hexdigest()}_master'
# variables end

def create_new_qr(pswrd, name):
    qr = qrcode.make(pswrd)
    os.makedirs('qr', exist_ok=True)
    qr.save(f'qr/{name}.png')

create_new_qr(QR_PASSWORD, 'code')
create_new_qr(QR_MASTER_PASSWORD, 'master_code')

cap = cv2.VideoCapture(0)

face_detector = cv2.FaceDetectorYN.create('model/face/face_detection_yunet_120x160.onnx', '', (160, 120))
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
okay_count = 0
okay_cause = ''

while True:
    okay_flag = False
    outs = []
    ret, img = cap.read()

    # img = imutils.resize(frame, width=764)
    clone = copy.deepcopy(img)
    w, h = [160, 120]


    image_width, image_height = img.shape[1], img.shape[0]
    debug_image = copy.deepcopy(img)


    face_detector.setInputSize(img.shape[:2][::-1])
    _, faces = face_detector.detect(img)
    faces = faces if faces is not None else []

    """
    if len(fs):
        af = face_recognizer.alignCrop(img, fs[0])
        cv2.imwrite('test.jpg', af)
    """

    for face in faces:
        
        aligned_face = face_recognizer.alignCrop(img, face)
        face_feature = face_recognizer.feature(aligned_face)
        outs.append( aligned_face )


        result, user = match(face_feature)
        
        id, score = user if result else ("?", 0.0)
        color = (80, 230, 90) if result else (0, 0, 255)

        if score > COSINE_THRESHOLD:
            okay_flag = True
            okay_cause = f'User {id}'


        text = "{} ({:.2f})".format(id, score)

        box = list(map(int, face[:4]))
        color = (0, 255, 0) if result else (0, 0, 255)
        thickness = 2
        cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)

        cv2.putText(img, text, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)


    isQ, q_decoded, q_pos, _ = qcd_detector.detectAndDecodeMulti(debug_image)
    if isQ:
        if not QR_PASSWORD == get_hash_pass():
            QR_PASSWORD = get_hash_pass()
            create_new_qr(QR_PASSWORD, 'code')
        for s, p in zip(q_decoded, q_pos):
            if s in [QR_PASSWORD, QR_MASTER_PASSWORD]:
                color = (80, 230, 90)
                txt = 'Valid'
                okay_flag = True
                okay = PASS_SCORE
                okay_cause = 'QR'.ljust(11)
            elif not s:
                color = (0, 0, 255)
                txt = ''
            else:
                color = (0, 0, 255)
                txt = 'Invalid'
            cv2.polylines(img, [p.astype(int)], True, color, 2)
            cv2.putText(img, txt, p[0].astype(int),
                      cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2, cv2.LINE_AA)

    if okay_flag: okay += 1
    else: okay = 0

    if okay >= PASS_SCORE:
        okay_count += 1
        print(f'Pass : {str(okay_count).ljust(11)}\nCause: {okay_cause}\n\033[2A', end="")
        okay = 0

    cv2.imshow("Face Detection", img)
    k = cv2.waitKey(1) & 0xff
    #if -1 < k: print(k)
    if k == ord('s'):
        os.makedirs('faces/_img', exist_ok=True)
        cpath = lambda n, d, ext: f"faces/{d}{str(n).zfill(6)}{ext}"
        i = 0
        for out in outs:
            while True:
                i += 1
                if not os.path.exists( cpath(i, '', '.npy') ):
                    n = cpath(i, '', '')
                    ft = face_recognizer.feature(out)
                    np.save(n, ft)
                    print(f'> {n}.npy')
                    print(ft)
                    print('=' * 10)

                    cv2.imwrite(cpath(i, '_img/', '.jpg'), out)
                    break
        FEATURES = load_features()
    elif k == ord('q') or k == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()

print('')
