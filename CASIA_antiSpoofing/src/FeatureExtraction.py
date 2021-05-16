import cv2
import numpy as np
import os
from skimage import feature
from sklearn.externals import joblib

class LBP:
    def __init__(self, p, r):
        # store the number of points p and radius r
        self.p = p
        self.r = r

    def getLBPH(self, image, eps=1e-7):
        # compute the LBP representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.p, self.r, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.p + 3), range=(0, self.p + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the LBPH
        return hist, lbp


def getImageQuadrants(img):
    Q1 = img[0:img.shape[0] // 2, img.shape[1] // 2:img.shape[1]]
    Q2 = img[0:img.shape[0] // 2, 0:img.shape[1] // 2]
    Q3 = img[img.shape[0] // 2:img.shape[0], 0:img.shape[1] // 2]
    Q4 = img[img.shape[0] // 2:img.shape[0], img.shape[1] // 2:img.shape[1]]
    return (Q1, Q2, Q3, Q3, Q4)


def getFeatureVector(face_img, lbp):
    (Q1, Q2, Q3, Q3, Q4) = getImageQuadrants(face_img)
    (Q11, Q12, Q13, Q13, Q14) = getImageQuadrants(Q1)
    (Q21, Q22, Q23, Q23, Q24) = getImageQuadrants(Q2)
    (Q31, Q32, Q33, Q33, Q34) = getImageQuadrants(Q3)
    (Q41, Q42, Q43, Q43, Q44) = getImageQuadrants(Q4)
    face_img_lbph_Q12, _ = lbp.getLBPH(Q12)
    face_img_lbph_Q11, _ = lbp.getLBPH(Q11)
    face_img_lbph_Q14, _ = lbp.getLBPH(Q14)
    face_img_lbph_Q13, _ = lbp.getLBPH(Q13)

    face_img_lbph_Q22, _ = lbp.getLBPH(Q22)
    face_img_lbph_Q21, _ = lbp.getLBPH(Q12)
    face_img_lbph_Q24, _ = lbp.getLBPH(Q24)
    face_img_lbph_Q23, _ = lbp.getLBPH(Q23)

    face_img_lbph_Q32, _ = lbp.getLBPH(Q32)
    face_img_lbph_Q31, _ = lbp.getLBPH(Q31)
    face_img_lbph_Q34, _ = lbp.getLBPH(Q34)
    face_img_lbph_Q33, _ = lbp.getLBPH(Q33)

    face_img_lbph_Q42, _ = lbp.getLBPH(Q42)
    face_img_lbph_Q41, _ = lbp.getLBPH(Q41)
    face_img_lbph_Q44, _ = lbp.getLBPH(Q44)
    face_img_lbph_Q43, _ = lbp.getLBPH(Q43)

    combined_face_lbph = np.hstack((face_img_lbph_Q11, face_img_lbph_Q12, face_img_lbph_Q13, face_img_lbph_Q14,
                                    face_img_lbph_Q21, face_img_lbph_Q22, face_img_lbph_Q23, face_img_lbph_Q24,
                                    face_img_lbph_Q31, face_img_lbph_Q32, face_img_lbph_Q33, face_img_lbph_Q34,
                                    face_img_lbph_Q41, face_img_lbph_Q42, face_img_lbph_Q43, face_img_lbph_Q44))

    return combined_face_lbph