import cv2
import numpy as np
from sklearn.externals import joblib
import dlib
from dlib import rectangle
import FeatureExtraction as ft

cap = cv2.VideoCapture(0)#('../data/CASIA-FA/train_release/3/HR_2.avi')
classifier = joblib.load('../data/rf_FASD.pkl')
predictorPath = "../faceModels/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)


def rectT0BoundingBox(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def check_for_liveness(img, detector):
    faces = detector(img)
    if len(faces) > 0 and img is not None:
        for face in faces:
            (x, y, w, h) = rectT0BoundingBox(face)
            print('-----------')
            print(x, y, w, h)
            print('-----------')
            if x >= 0 and y >= 0 and w <= img.shape[0] and h <= img.shape[1]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                gray = cv2.cvtColor(img[x:x + w, y:y + h], cv2.COLOR_BGR2GRAY)#cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                lbp = ft.LBP(8, 1)
                lbph = ft.getFeatureVector(gray, lbp)
                lbph = lbph.reshape(1, -1)
                prediction_value = classifier.predict(lbph)
                prediction = ''
                if prediction_value == 0:
                    prediction = 'FAKE'
                elif prediction_value == 1:
                    prediction = 'REAL'
                cv2.putText(img, prediction+': '+str(max(classifier.predict_proba(lbph))), (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, .4, (255, 255, 255))#(x, (y + h) - 10)
    return img

while True:
    image = cap.read()[1]
    if not cap.read()[0]:
        break  # continue
    image = cv2.flip(image, 1)
    i = check_for_liveness(image.copy(), detector)
    cv2.imshow('Frame', image)
    cv2.imshow('FrameDetection', i)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()