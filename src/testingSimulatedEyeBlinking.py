import dlib
import cv2
import numpy as np
from dlib import rectangle

predictorPath = "../faceModels/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

def extractFacialLandmarks(faceROI, img, imgScale, predictor):
    upscaledFaceROI = rectangle(int(faceROI.left() / imgScale), int(faceROI.top() / imgScale),
                                int(faceROI.right() / imgScale), int(faceROI.bottom() / imgScale))


    # predict facial landmark points
    facialLandmarks = predictor(img, upscaledFaceROI)

    # make an array of the landmark points with 68 (x,y) coordinates
    facialLandmarkCoords = np.array([[p.x, p.y] for p in facialLandmarks.parts()])

    # transpose the landmark points so that we deal with a 2xn and not an nx2 model, it makes
    # calculations easier along the way when its a row for x's and a row for y's
    return facialLandmarkCoords.T

def downScaleImg(img, imgScale, maxImgSizeForDetection):
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    return scaledImg, imgScale

def getFacialLandmarks(textureImage, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    downScaledImg, imgScale = downScaleImg(textureImage, imgScale, maxImgSizeForDetection)

    # detect face on smaller image (much faster)
    detectedFacesROI = detector(downScaledImg, 1)

    # return nothing if no faces are found
    if len(detectedFacesROI) == 0:
        return None

    # list of facial landmarks for each face in the mapped image
    facialLandmarksList = []
    for faceROI in detectedFacesROI:
        facialLandmarks = extractFacialLandmarks(faceROI, textureImage, imgScale, predictor)
        facialLandmarksList.append(facialLandmarks)
    # return list of faces
    return facialLandmarksList

def reshape_for_polyline(array):
    # do not know what the outer dimension is, but make it an 1x2 now
    return np.array(array, np.int32).reshape((-1, 1, 2))

def drawImposterLandmarks(frame, landmarks, black_image):
    #black_image = np.zeros(frame.shape, np.uint8)

    landmarks = landmarks.T
    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[30:35])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])

    color = (255, 255, 255)
    thickness = 3

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image

cap = cv2.VideoCapture(0)
while True:
    image = cap.read()[1]
    image = cv2.flip(image, 1)
    faces = getFacialLandmarks(image, detector, predictor, 320)
    black_image = np.zeros(image.shape, np.uint8)
    another_black_image = np.zeros(image.shape, np.uint8)
    left_eye_frame = np.zeros(image.shape, np.uint8)
    right_eye_frame = np.zeros(image.shape, np.uint8)
    nose_frame = np.zeros(image.shape, np.uint8)
    mouth_frame = np.zeros(image.shape, np.uint8)
    if faces is not None:

        for facialLandmarks2D in faces:
            # draw the landmarks of all the faces detected from the source frame
            black_image = drawImposterLandmarks(image, facialLandmarks2D, black_image)

            left_eye = facialLandmarks2D.T[42:48]
            center_of_left_eye = (int(1 / 2 * (np.max(left_eye[:, 0]) + np.min(left_eye[:, 0]))),
                                  int(1 / 2 * (np.max(left_eye[:, 1]) + np.min(left_eye[:, 1]))))

            right_eye = facialLandmarks2D.T[36:42]
            center_of_right_eye = (int(1 / 2 * (np.max(right_eye[:, 0]) + np.min(right_eye[:, 0]))),
                                  int(1 / 2 * (np.max(right_eye[:, 1]) + np.min(right_eye[:, 1]))))

            outer_lip = facialLandmarks2D.T[48:60]
            center_of_mouth = (int(1 / 2 * (np.max(outer_lip[:, 0]) + np.min(outer_lip[:, 0]))),
                                  int(1 / 2 * (np.max(outer_lip[:, 1]) + np.min(outer_lip[:, 1]))))

            nose_ridge = facialLandmarks2D.T[27:31]
            center_of_nose_ridge = (int(1 / 2 * (np.max(nose_ridge[:, 0]) + np.min(nose_ridge[:, 0]))),
                               int(1 / 2 * (np.max(nose_ridge[:, 1]) + np.min(nose_ridge[:, 1]))))
            lower_nose = facialLandmarks2D.T[30:35]
            center_of_lower_nose = (int(1 / 2 * (np.max(lower_nose[:, 0]) + np.min(lower_nose[:, 0]))),
                               int(1 / 2 * (np.max(lower_nose[:, 1]) + np.min(lower_nose[:, 1]))))

            center_of_nose = ((center_of_lower_nose[0] + center_of_nose_ridge[0]) // 2,
                           (center_of_lower_nose[1] + center_of_nose_ridge[1]) // 2)

            left_eye_OutLine = cv2.convexHull(left_eye)
            right_eye_OutLine = cv2.convexHull(right_eye)
            face_Outline = cv2.convexHull(facialLandmarks2D.T)
            cv2.fillConvexPoly(another_black_image, face_Outline, (255, 255, 255))
            masked_face = cv2.bitwise_and(image, another_black_image)
            masked_face_copy = cv2.bitwise_and(image, another_black_image)

            cv2.circle(masked_face, center_of_left_eye, 1, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_left_eye, 20, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_right_eye, 1, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_right_eye, 20, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_mouth, 1, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_mouth, 35, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_nose, 1, (0, 0, 255), 1)
            cv2.circle(masked_face, center_of_nose, 35, (0, 0, 255), 1)

            cv2.circle(masked_face_copy, center_of_left_eye, 20, (0, 0, 255), -1)
            cv2.circle(masked_face_copy, center_of_right_eye, 20, (0, 0, 255), -1)
            cv2.circle(masked_face_copy, center_of_mouth, 35, (0, 0, 255), -1)
            cv2.circle(masked_face_copy, center_of_nose, 35, (0, 0, 255), -1)

            cv2.imshow('try', masked_face)
            cv2.imshow('try try', masked_face_copy)

    cv2.imshow('something', image)
    cv2.imshow('landmarks', black_image)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()