import numpy as np
import cv2
import statistics
# here src is the image from which the pixels will be pasted on to the dst image


def findROI(src, dst, mask):
    # index all of the pixels in the mask defining our region of interest
    # i.e the black pixels
    # maskIndices is a tuple of 2 arrays with row and column indices respectively
    roiIdxs = np.where(mask == 0)

    # use the mask indices to isolate the region of interest in both the src and dst image
    roiSrc = src[roiIdxs]
    roiDst = dst[roiIdxs]

    return roiSrc, roiDst, roiIdxs

def transferComlexion(src, dst, mask):
    # ensure the dst image with object being transfered has a numpy copy of itself for use
    # changes will be applied to the copy
    recolouredDst = np.copy(dst)
    # locate all the pixels in corresponding to mask from both src and dst
    # roiIdxs are returned because we still have more use for them
    roiSrc, roiDst, roiIdxs = findROI(src, dst, mask)

    # find the average bgr values for both src and dst regions of interest
    # avgSrcBGR, avgDstBGR = findAverageColours(roiSrc, roiDst)
    # find the average BGR values for the src and dst regions of interest
    avgSrcBGR = np.mean(roiSrc, axis=0)
    # print(avgSrcBGR.shape)
    avgDstBGR = np.mean(roiDst, axis=0)

    # recolouredDst[roiIdxs] = applyColourAverages(roiDst, avgSrcBGR, avgDstBGR)
    # remove the general colour from the dst region of interest
    roiDst = roiDst - avgDstBGR
    # add the general colour of the src region of interest to the dst region of interest
    roiDst = roiDst + avgSrcBGR
    # make sure the BGR values are within range
    roiDst = np.clip(roiDst, 0, 255)
    # paste the newly coloured region of interest onto the numpy copy of the dst image
    recolouredDst[roiIdxs] = roiDst

    return recolouredDst

#Color Transfer between Images by Reinhard et al, 2001.
def improvedTransferComlexion(src, dst, mask):
    # convert src and dst to L*a*b* colour space
    srcLAB = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    dstLAB = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)

    # ensure the dst image with object being transfered has a numpy copy of itself for use
    # changes will be applied to the copy
    recolouredDst = np.copy(dstLAB)
    #print('recolouredDst', recolouredDst.shape)

    # locate all the pixels in corresponding to mask from both src and dst
    # roiIdxs are returned because we still have more use for them
    roiSrc, roiDst, roiIdxs = findROI(srcLAB, dstLAB, mask)
    '''
    # convert src and dst to L*a*b* colour space
    #roiSrcLAB = cv2.cvtColor(src, cv2.COLOR_RGB2LAB, src)
    #roiDstLAB = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB, dst)

    # isolate each channel of both src and dst
    # for roiSrc
    roiSrc_L = roiSrc[:, 0].astype('float64')
    roiSrc_A = roiSrc[:, 1].astype('float64')
    roiSrc_B = roiSrc[:, 2].astype('float64')

    # for roiDst
    roiDst_L = roiDst[:, 0].astype('float64')
    roiDst_A = roiDst[:, 1].astype('float64')
    roiDst_B = roiDst[:, 2].astype('float64')

    # obtain the standard deviation and mean of each channel in both src and dst
    # src standard deviation
    roiSrc_L_std = np.std(roiSrc_L, axis=0)#statistics.stdev(roiSrc_L)
    roiSrc_A_std = np.std(roiSrc_A, axis=0)#statistics.stdev(roiSrc_A)
    roiSrc_B_std = np.std(roiSrc_B, axis=0)#statistics.stdev(roiSrc_B)

    # dst standard deviation
    roiDst_L_std = np.std(roiDst_L, axis=0)#statistics.stdev(roiDst_L)
    roiDst_A_std = np.std(roiDst_A, axis=0)#statistics.stdev(roiDst_A)
    roiDst_B_std = np.std(roiDst_B, axis=0)#statistics.stdev(roiDst_B)

    # src mean
    roiSrc_L_mean = np.mean(roiSrc_L, axis=0)#statistics.mean(roiSrc_L)
    roiSrc_A_mean = np.mean(roiSrc_A, axis=0)#statistics.mean(roiSrc_A)
    roiSrc_B_mean = np.mean(roiSrc_B, axis=0)#statistics.mean(roiSrc_B)

    # src mean
    roiDst_L_mean = np.mean(roiDst_L, axis=0)#statistics.mean(roiDst_L)
    roiDst_A_mean = np.mean(roiDst_A, axis=0)#statistics.mean(roiDst_A)
    roiDst_B_mean = np.mean(roiDst_B, axis=0)#statistics.mean(roiDst_B)

    # subtract the means from src
    roiSrc_L -= roiSrc_L_mean
    roiSrc_A -= roiSrc_A_mean
    roiSrc_B -= roiSrc_B_mean

    # scale by the standard deviations
    # l = (lStdDst / lStdSrc) * l
    roiDst_L *= roiSrc_L_std/roiDst_L_std
    roiDst_A *= roiSrc_A_std/roiDst_A_std
    roiDst_B *= roiSrc_B_std/roiDst_B_std

    # add the means of the src
    roiDst_L += roiSrc_L_mean
    roiDst_A += roiSrc_A_mean
    roiDst_B += roiSrc_B_mean

    # ensure values are within colour range
    #roiDst_L = np.clip(roiDst_L, 0, 255)
    #roiDst_A = np.clip(roiDst_A, 0, 255)
    #roiDst_B = np.clip(roiDst_B, 0, 255)

    # merge the channels together and convert back to the BGR color space,
    # being sure to utilize the 8-bit unsigned integer data type
    print(roiDst.shape)
    #roiDst = cv2.merge([roiDst_L, roiDst_A, roiDst_B])
    roiDst[:, 0] = roiDst_L
    roiDst[:, 1] = roiDst_A
    roiDst[:, 2] = roiDst_B

    # ensure values are within colour range
    roiDst = np.clip(roiDst, 0, 255)

    recolouredDst[roiIdxs] = roiDst.astype('uint8')
    #print(roiDst.shape)
    # convert back to BGR colour space
    #roiDst = cv2.inRange(roiDst, 0, 255).astype('uint8')
    #print(roiDst.shape)

    #roiDst = cv2.cvtColor(roiDst, cv2.COLOR_LAB2BGR)
    recolouredDst = cv2.cvtColor(recolouredDst, cv2.COLOR_LAB2BGR)
    cv2.imshow('qwerty', recolouredDst)
    # paste the newly coloured region of interest onto the numpy copy of the dst image
    #recolouredDst[roiIdxs] = roiDst
    '''

    # find the average bgr values for both src and dst regions of interest
    # avgSrcBGR, avgDstBGR = findAverageColours(roiSrc, roiDst)
    # find the average BGR values for the src and dst regions of interest
    avgSrcLAB = np.mean(roiSrc, axis=0)
    # print(avgSrcBGR.shape)
    avgDstLAB = np.mean(roiDst, axis=0)

    # find the standard deviation BGR values for the src and dst regions of interest
    stdevSrcLAB = np.std(roiSrc, axis=0)
    #print(stdevSrcLAB)
    stdevDstLAB = np.std(roiDst, axis=0)

    # recolouredDst[roiIdxs] = applyColourAverages(roiDst, avgSrcBGR, avgDstBGR)
    # remove the general colour from the dst region of interest
    roiDst = roiDst - avgDstLAB
    # scale by the standard deviations
    roiDst = 1/(stdevDstLAB/stdevSrcLAB)*roiDst

    # add the general colour of the src region of interest to the dst region of interest
    roiDst = roiDst + avgSrcLAB
    # make sure the BGR values are within range
    roiDst = np.clip(roiDst, 0, 255)
    # paste the newly coloured region of interest onto the numpy copy of the dst image
    recolouredDst[roiIdxs] = roiDst
    recolouredDst = cv2.cvtColor(recolouredDst, cv2.COLOR_LAB2BGR)

    return recolouredDst



# feather amount is a percentage, for controlling the size of the area, which will be subjected to weights
# weights is a percentage, for controlling the amount of colour taken from the src and dst images respectively
def pasteAndBlendNewFace(src, dst, mask, featherAmount=0.2):
    # locate all the pixels in corresponding to mask from both src and dst
    # roiIdxs are returned because we still have more use for them
    roiSrc, roiDst, roiIdxs = findROI(src, dst, mask)

    # making it a 2 column array with (x,y) index pairs for the non-white pixels
    roiCoords = np.hstack((roiIdxs[1][:, np.newaxis], roiIdxs[0][:, np.newaxis]))
    # listing the distances between points enclosed by the ROI and the face outlines
    roiToOutlineDistances = determineROIToOutlineDistances(roiCoords)
    # feather amount as a function of the size of the region of interest
    featherAmount = determineFeatheringAmount(featherAmount, roiCoords)
    # ensure that the weight values for the pixel 'intensity' are between 0 and 1
    # and that they are a function of the distance between a pixel and the enclosing face outline
    weights = np.clip(roiToOutlineDistances / featherAmount, 0, 1)
    # a numpy copy of the dst image with the user's face
    mergedImg = np.copy(dst)
    mergedButNotBlendedImg = mergedImg.copy()
    # a portion of the dst region of interest will have its pixels be the src pixels
    # and the complement of that will be the pixels belonging to the dst ROI
    # here is the formula: D(roi) = w*S(roi) + w'*D(roi)
    mergedImg[roiIdxs] = weights[:, np.newaxis] * roiSrc + (1 - weights[:, np.newaxis]) * roiDst
    mergedButNotBlendedImg[roiIdxs] = roiSrc
    #cv2.imshow("merged", mergedImg)
    #cv2.imshow("mergedButNotBlendedImg", mergedButNotBlendedImg)
    return mergedImg, mergedButNotBlendedImg

def determineROIToOutlineDistances(roiCoords):
    # find outline around face
    # (uses snake energy equations...)
    faceOutLine = cv2.convexHull(roiCoords)

    # get the distance of every point to the outline in an array
    distances = np.zeros(roiCoords.shape[0])
    for i, roiC in enumerate(roiCoords):#range(roiCoords.shape[0]):
        # finding shortest distance between roiCoords and the face outline
        distances[i] = cv2.pointPolygonTest(faceOutLine, tuple(roiC), True)

    return distances

def determineFeatheringAmount(featheringAmount, roiCoords):
    # establishing the size of the face, longitudinally and latitudinally
    faceSize = np.max(roiCoords, axis=0) - np.min(roiCoords, axis=0)

    # feathering amount as a portion of the face size
    featheringAmount *= np.max(faceSize)

    return featheringAmount

def findAverageColours(roiSrc, roiDst):
    # find the average BGR values for the src and dst regions of interest
    avgSrcBGR = np.mean(roiSrc, axis=0)
    #print(avgSrcBGR.shape)
    avgDstBGR = np.mean(roiDst, axis=0)
    #print(avgSrcBGR.shape)

    return avgSrcBGR, avgDstBGR

def applyColourAverages(roiDst, avgSrcBGR, avgDstBGR):
    # remove the general colour from the dst region of interest
    roiDst = roiDst - avgDstBGR
    # add the general colour of the src region of interest to the dst region of interest
    roiDst = roiDst + avgSrcBGR
    # make sure the BGR values are within range
    roiDst = np.clip(roiDst, 0, 255)

    return roiDst