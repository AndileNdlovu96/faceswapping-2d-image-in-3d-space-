import os
import cv2
import  numpy as np
import time
import dlib
from ProjectionModel import ProjectionModel

from dlib import rectangle

from Optimizer import GNA

import FaceRenderer
import FaceProcessor




def main():
    data_dir = '../data/'
               #utkface_dataset/'
    imposters = os.listdir(data_dir)
    imgTarget = cv2.imread('../data/AndileIDPhoto.jpg')

    # loading the keypoint detection model, the image and the 3D model
    # ----------------------------------------------------------------
    # load face landmark prediction model from here:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictorPath = "../faceModels/shape_predictor_68_face_landmarks.dat"

    # initialize face detector and face landmark estimator
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorPath)

    # load the 3D face model from numpy file
    # candide 3D face model from here:
    # http://www.icg.isy.liu.se/candide/
    faceModelFile = np.load("../faceModels/candide.npz")
    mean3DShape = faceModelFile["mean3DShape"]
    blendShapes = faceModelFile["blendshapes"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    # mesh must be inspected and modified where faulty before use
    mesh = faceModelFile["mesh"]
    # order mesh winding
    mesh = orderMeshWinding(mesh, mean3DShape)
    # initialize the projection model describing the 3D object
    # to be rendered
    projectionModel = ProjectionModel(blendShapes.shape[0])

    print(len(imposters))  # 3252 images
    for imposter in imposters:
        if imposter[-3:].lower() != 'jpg':
            if imposter[-3:].lower() != 'png':
                continue
        imgSourcePath = data_dir+'/'+imposter
        imgSource = cv2.imread(imgSourcePath)
        #print(imposter)
        source_frame = imgSource

        # obtain the texture coordinates of the 2D projection of the 3D model of the target face
        textureCoords = getTextureCoordinates(imgTarget, mean3DShape, blendShapes, idxs3D, idxs2D,
                                              detector,
                                              predictor)

        # apply this recieved texture coordinates to the image recieved from the webcam
        # generate model on game engine and modify parameters as looping occures
        # this is done so that we need  not load textures from the beginning
        # when rendering the face stolen by the imposter
        faceswapper = FaceRenderer.FaceSwapper(imgSource, imgTarget, textureCoords, mesh)

        # locate any faces on the frame
        faces = getFacialLandmarks(imgSource, detector, predictor, 320)

        # if any faces are detected, this is where we swap them...
        if faces is not None:
            black_image = np.zeros(imgSource.shape, np.uint8)
            for facialLandmarks2D in faces:
                # draw the landmarks of all the faces detected from the source frame
                onlyLandmarks = drawImposterLandmarks(imgSource, facialLandmarks2D, black_image)
                X = [mean3DShape[:, idxs3D], blendShapes[:, :, idxs3D]]
                Y = facialLandmarks2D[:, idxs2D]

                # we are going to use the 46 common vertices in on both
                # the mean 3D shape and the 2D face landmark points
                Betas = projectionModel.getInitialBetas(X[0], Y)
                # print('Betas', Betas)
                Betas = GNA(Betas, projectionModel.residual, projectionModel.J, X, Y, showPerformance=False)

                # rendering the model to an image

                # assemble the 3D model of face with appropriate beta parameters
                # scale value
                a = Betas[0]

                # rotation values
                # xyz rotation
                r = Betas[1:4]

                # translation values
                # xy translation
                t = Betas[4:6]

                # blendshape weights
                w = Betas[6:]

                # mean3DShape - the neutral face
                S_0 = mean3DShape

                # the blendshapes - various facial expressions
                S_i_to_n = blendShapes

                # rotation matrix from the rotation vector, Rodriguez pattern
                # this makes it possibe to apply rotation to coordinates by
                # matrix multiplication
                # r1,\
                R, _ = cv2.Rodrigues(r)

                # just the x and y rows
                # R = r1[:2]

                # allowing for operator broadcasting R1 to R3 by adding new axes
                W_i_to_n = w[:, np.newaxis, np.newaxis]

                # S= S_0 + W_i_to_n*S_i_to_n --> this is the summed faces
                S = S_0 + np.sum((W_i_to_n * S_i_to_n), axis=0)

                # allowing for operator broadcasting R1 to R2 by adding new axis
                T = t[:, np.newaxis]

                # projectedFace = a*R.S + t
                # only apply the xy translation matrix T to the xy coords of 3d face,
                # not z as well
                projected3DFace = a * np.dot(R, S)
                projected3DFace[:2, :] = (a * np.dot(R, S)[:2, :]) + T

                # use projected 3D model and add textures onto it
                # and obtain an image percieved through viewpoint
                # this is the 3D rendered target face
                targetFaceIn3D = faceswapper.renderFace(projected3DFace)

                mask = np.ones(targetFaceIn3D.shape[:2])
                mask = cv2.inRange(targetFaceIn3D, 0, 0)
                #cv2.imshow('mask', mask)

                # targetFaceIn3D_WithComplexionImg = FaceProcessor.improvedTransferComlexion(self.imgSource, targetFaceIn3D,
                # mask)
                targetFaceIn3D_WithComplexionImg = FaceProcessor.transferComlexion(imgSource, targetFaceIn3D, mask)
                #cv2.imshow('targetFaceIn3D_WithComplexionImg', targetFaceIn3D_WithComplexionImg)
                #cv2.imshow('targetFaceIn3D', targetFaceIn3D)
                #cv2.imshow('onlyLandmarks', onlyLandmarks)
                imgSource, imgSourceNotBlended = FaceProcessor.pasteAndBlendNewFace(targetFaceIn3D_WithComplexionImg, imgSource, mask)

                x = '../data/Results/'+imposter[:-4]+ '/'
                os.makedirs(x, exist_ok=True)
                cv2.imwrite(x +imposter[:-4]+'__'+"imposter"+".jpg", source_frame)
                cv2.imwrite(x +imposter[:-4]+'__'+"target"+".jpg", imgTarget)
                cv2.imwrite(x + imposter[:-4]+'__'+"landmarks"+".jpg", onlyLandmarks)
                cv2.imwrite(x + imposter[:-4]+'__'+"mask"+".jpg", cv2.bitwise_not(src=mask, mask=cv2.inRange(np.zeros(black_image.shape, np.uint8), 0,0)))
                cv2.imwrite(x + imposter[:-4]+'__'+"targetFace_no_complexion"+".jpg", targetFaceIn3D)
                cv2.imwrite(x + imposter[:-4]+'__'+"targetFace_with_complexion"+".jpg", targetFaceIn3D_WithComplexionImg)
                cv2.imwrite(x + imposter[:-4]+'__'+"result_no_blending"+".jpg", imgSourceNotBlended)
                cv2.imwrite(x + imposter[:-4]+'__'+"result_with_blending"+".jpg", imgSource)


    key = cv2.waitKey(1)

    if key == 27:
        # cap.release()
        cv2.destroyAllWindows()



def orderMeshWinding(mesh, mean3DShape):
    for m in mesh:
        triangleIdxs = m
        triangleFace = mean3DShape[:, triangleIdxs]

        # obtain the vertices of each triangle face
        #normal = getNormal(triangleFace)
        A = triangleFace[0]
        B = triangleFace[1]
        C = triangleFace[2]

        # (B-A)X(C-A) = N
        normalVector = np.cross(B - A, C - A, axis=0)
        # n^ = (1/|N|)N
        normalUnitVector = normalVector / np.linalg.norm(normalVector)

        # if z coordinate of the face is negative, flip the meshing order
        if normalUnitVector[2] > 0:
            # reverse the winding order
            #m = reverseMeshWinding(triangleFace)
            m = [triangleFace[1], triangleFace[0], triangleFace[2]]

    return mesh

def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    normalVector = np.cross(b - a, c - a, axis=0)
    normalUnitVector = normalVector / np.linalg.norm(normalVector)

    return normalUnitVector

def reverseWinding(triangle):
    # reverse the winding order
    return [triangle[1], triangle[0], triangle[2]]

def fit3DModelTo2DlandMarks(facialLandmarks2D, mean3DShape, blendshapes, idxs2D, idxs3D, textextureImagePM):
    X = [mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]]
    Y = facialLandmarks2D[:, idxs2D]
    # we are going to use the 46 common vertices in on both the mean 3D shape and the 2D face landmark points
    Beta = textextureImagePM.getInitialBetas(X[0], Y)
    #print('Beta', Beta)
    Beta = GNA(Beta, textextureImagePM.residual, textextureImagePM.J, X, Y, showPerformance=False)
    # acquire the 3D landmark coordinates
    landmarkCoordinates = textextureImagePM.f([mean3DShape, blendshapes], Beta)

    return landmarkCoordinates


def extractFacialLandmarks(faceROI, img, imgScale, predictor):
    # to scale down, the image was multiplied by imgScale(a fraction), and now that must be undone
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
    # look at biggest dimension of image
    # if this is greater than the biggest image size that the detector can handle,
    # then ensure that the image is made smaller than the detectable image size
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

def getTextureCoordinates(textureImage, mean3DShape, blendShapes, idxs3D, idxs2D, detector, predictor):
    textureImagePM = ProjectionModel(blendShapes.shape[0])
    # get on;y the first face in the list of facial landmarks. There is no implemenntation on dealing with multiple faces
    facialLandmarks2D = getFacialLandmarks(textureImage, detector, predictor)[0]

    textureCoordinates = fit3DModelTo2DlandMarks(facialLandmarks2D, mean3DShape, blendShapes, idxs2D, idxs3D, textureImagePM)

    return textureCoordinates


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

if __name__ == '__main__':
    main()