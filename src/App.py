import dlib
import cv2
import numpy as np

from ProjectionModel import ProjectionModel

from dlib import rectangle

from Optimizer import GNA

import FaceRenderer
import FaceProcessor

import UserInterface


def main():
    # loading the keypoint detection model, the image and the 3D model
    # ----------------------------------------------------------------
    # load face landmark prediction model from here:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictorPath = "../faceModels/shape_predictor_68_face_landmarks.dat"
    # load face image to map onto webcam frame
    # my own personal image...
    # the sited for other pictures available in the data folder:
    # amber rose: large.jpg -> https://weheartit.com/entry/25343067/large.jpg
    # rihanna: large_fustany-beauty-makeup-rihanna_s_makeup_looks-7_copy.jpg ->
    # https://static.fustany.com/images/en/photo/large_fustany-beauty-makeup-rihanna_s_makeup_looks-7_copy.jpg rihanna

    faceImagePath = "../data/jolie.jpg"#"../data/AndileIDPhoto.jpg"

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

    # obtain the image with the face one desires to steal
    textureImage = cv2.imread(faceImagePath)
    # turn on buit-in webcam
    cap = cv2.VideoCapture(0)
    # give webcam some time to warm up
    #time.sleep(2.0)
    # set the output path for saving modified images
    #outputPath = 'C:\\Users\\Andile Ndlovu\\Desktop'

    # obtain the real-time feed of the face of the imposter
    _, cameraImage = cap.read()
    # this is just so one can track his own movements on the cam
    cameraImage = cv2.flip(cameraImage, 1)

    # obtain the texture coordinates of the 2D projection of the 3D model of the target face
    textureCoords = getTextureCoordinates(textureImage, mean3DShape, blendShapes, idxs3D, idxs2D, detector, predictor)
    #print('textureCoords.shape', textureCoords.shape)

    # apply this recieved texture coordinates to the image recieved from the webcam
    # generate model on game engine and modify parameters as looping occures
    # this is done so that we need  not load textures from the beginning
    # when rendering the face stolen by the imposter
    faceswapper = FaceRenderer.FaceSwapper(cameraImage, textureImage, textureCoords, mesh)

    while True:
        # capture a frame from the webcam
        _, cameraImage = cap.read()
        cameraImage = cv2.flip(cameraImage, 1)
        # locate any faces on the frame
        faces = getFacialLandmarks(cameraImage, detector, predictor, 320)
        #print('faces', len(faces))
        # if any faces are detected, this is where we swap them...
        if faces is not None:
            for facialLandmarks2D in faces:
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
                #r1,\
                R, _ = cv2.Rodrigues(r)

                # just the x and y rows
                #R = r1[:2]

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
                newFaceImg = faceswapper.renderFace(projected3DFace)

                mask = np.ones(newFaceImg.shape[:2])
                mask = cv2.inRange(newFaceImg, 0, 0)
                #cv2.imshow('mask', mask)

                newFaceWithComplexionImg = FaceProcessor.improvedTransferComlexion(cameraImage, newFaceImg, mask)
                #newFaceWithComplexionImg = FaceProcessor.transferComlexion(cameraImage, newFaceImg, mask)
                #cv2.imshow('newFaceWithComplexionImg', newFaceWithComplexionImg)
                cameraImage = FaceProcessor.pasteAndBlendNewFace(newFaceWithComplexionImg, cameraImage, mask)
                #cv2.imshow('new image', newCameraFace)
        cv2.imshow('Stolen Identity', cameraImage)

        key = cv2.waitKey(1)

        if key == 27:
            break
    cap.release()
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

def getTextureCoordinates(textureImage, mean3DShape, blendShapes, idxs3D, idxs2D, detector, predictor):
    textureImagePM = ProjectionModel(blendShapes.shape[0])
    # get on;y the first face in the list of facial landmarks. There is no implemenntation on dealing with multiple faces
    facialLandmarks2D = getFacialLandmarks(textureImage, detector, predictor)[0]

    textureCoordinates = fit3DModelTo2DlandMarks(facialLandmarks2D, mean3DShape, blendShapes, idxs2D, idxs3D, textureImagePM)

    return textureCoordinates

if __name__ == '__main__':
    main()