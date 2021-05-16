import dlib
import cv2
import numpy as np

from ProjectionModel import ProjectionModel

from dlib import rectangle

from Optimizer import GNA

import FaceRenderer
import FaceProcessor


from PIL import Image
from PIL import ImageTk

import datetime

import os
import time

import tkinter as tk
import threading



class UI:
    def __init__(self, cap, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.cap = cap
        self.outputPath = outputPath
        self.frame = None
        self.landmark_frame = None
        self.source_frame = None
        self.thread = None
        self.stopEvent = None
        self.isShowLandmarks = False
        self.isShowSource = False

        self.isVideoLoop = False
        self.isImgSource = True
        self.isImgTarget = False
        self.isSource_Frame = False
        self.imgSource = None
        self.imgTarget = None
        self.black_image = None

        self.imageSize = (500, 500)

        # initialize the root window and image panel
        self.root = tk.Tk()
        self.panel = None

        label = tk.Label()

        # create a button, that when pressed, will take the current
        # frame and save it to file
        btnSnap = tk.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
        btnSnap.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        #btnSnap.grid(row=2)

        # create a button, that when pressed, will take the current
        # frame and give the landmarks in detected in that frame
        #btnLand = tk.Button(self.root, text="See Landmarks!", command=self.showLandmarks)
        #btnLand.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)
        #btnLand.grid(row=1, column=0)


        # create a button, that when pressed, will take the current
        # frame and give the landmarks in detected in that frame
        #btnOrig = tk.Button(self.root, text="See Source!", command=self.showSource)
        #btnOrig.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)
        #btnOrig.grid(row=1, column=1)


        # create a button, that when pressed, will either alternate
        # to video mode or remain in image mode
        #btnVideoORImg = tk.Button(self.root, text="Video/Img mode", command=self.showVideoOrNot)
        #btnVideoORImg.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)

        # create a button, that when pressed, will take the current
        # frame and give the landmarks in detected in that frame
        btnResult = tk.Button(self.root, text="See Result!", command=self.showResultImg)
        btnResult.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)



        btnSource = tk.Button(self.root, text="See Imposter", command=self.showImposterImg)
        btnSource.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)

        btnTarget = tk.Button(self.root, text="See Target", command=self.showTargetImg)
        btnTarget.pack(side="bottom", fill="x", expand="yes", padx=10, pady=10)

        # start a thread that constantly pools the webcam for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.appLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("FACE STEALER!!!!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        try:

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

            faceImagePath = "../data/AndileIDPhoto.jpg"

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
            # time.sleep(2.0)
            # set the output path for saving modified images
            # outputPath = 'C:\\Users\\Andile Ndlovu\\Desktop'

            # obtain the real-time feed of the face of the imposter
            _, cameraImage = cap.read()
            # this is just so one can track his own movements on the cam
            cameraImage = cv2.flip(cameraImage, 1)

            # obtain the texture coordinates of the 2D projection of the 3D model of the target face
            textureCoords = getTextureCoordinates(textureImage, mean3DShape, blendShapes, idxs3D, idxs2D, detector,
                                                  predictor)
            # print('textureCoords.shape', textureCoords.shape)

            # apply this recieved texture coordinates to the image recieved from the webcam
            # generate model on game engine and modify parameters as looping occures
            # this is done so that we need  not load textures from the beginning
            # when rendering the face stolen by the imposter
            faceswapper = FaceRenderer.FaceSwapper(cameraImage, textureImage, textureCoords, mesh)



            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():  # this would be [while True:] without gui and threading complications
                # capture a frame from the webcam
                _, cameraImage = cap.read()
                cameraImage = cv2.flip(cameraImage, 1)
                self.source_frame = cameraImage
                # locate any faces on the frame
                faces = getFacialLandmarks(cameraImage, detector, predictor, 320)
                # print('faces', len(faces))
                # if any faces are detected, this is where we swap them...
                if faces is not None:
                    self.black_image = np.zeros(cameraImage.shape, np.uint8)
                    for facialLandmarks2D in faces:
                        # draw the landmarks of all the faces detected from the source frame
                        onlyLandmarks = drawImposterLandmarks(cameraImage, facialLandmarks2D, black_image)
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
                        newFaceImg = faceswapper.renderFace(projected3DFace)

                        mask = np.ones(newFaceImg.shape[:2])
                        mask = cv2.inRange(newFaceImg, 0, 0)
                        # cv2.imshow('mask', mask)

                        newFaceWithComplexionImg = FaceProcessor.improvedTransferComlexion(cameraImage, newFaceImg,
                                                                                           mask)
                        # newFaceWithComplexionImg = FaceProcessor.transferComlexion(cameraImage, newFaceImg, mask)
                        # cv2.imshow('newFaceWithComplexionImg', newFaceWithComplexionImg)
                        cameraImage = FaceProcessor.pasteAndBlendNewFace(newFaceWithComplexionImg, cameraImage, mask)
                        # cv2.imshow('new image', newCameraFace)

                        self.frame = cameraImage
                        print(self.frame.shape)
                        self.landmark_frame = onlyLandmarks
                        print(self.landmark_frame.shape)

                        if self.isShowLandmarks == True:
                            landmark_frame = cv2.resize(self.landmark_frame, None, fx=1 / 4, fy=1 / 4)
                            self.frame[0:self.frame.shape[0] * 1 // 4,
                            self.frame.shape[1] * 3 // 4: self.frame.shape[1], :] = landmark_frame

                        if self.isShowSource == True and self.isShowLandmarks == False:
                            source_frame = cv2.resize(self.source_frame, None, fx=1 / 4, fy=1 / 4)
                            self.frame[0:self.frame.shape[0] * 1 // 4,
                            self.frame.shape[1] * 3 // 4: self.frame.shape[1], :] = source_frame

                        elif self.isShowSource == True and self.isShowLandmarks == True:
                            source_frame = cv2.resize(self.source_frame, None, fx=1 / 4, fy=1 / 4)
                            self.frame[self.frame.shape[0] * 1 // 4:self.frame.shape[0] * 2 // 4,
                            self.frame.shape[1] * 3 // 4: self.frame.shape[1], :] = source_frame

                        # OpenCV represents images in BGR order; however PIL
                        # represents images in RGB order, so we need to swap
                        # the channels, then convert to PIL and ImageTk format
                        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = ImageTk.PhotoImage(image)

                        # if the panel is not None, we need to initialize it
                        if self.panel is None:
                            self.panel = tk.Label(image=image)
                            self.panel.image = image
                            self.panel.pack(side="top", padx=10, pady=10)
                            #self.panel.grid(row=0)

                        # otherwise, simply update the panel
                        else:
                            self.panel.configure(image=image)
                            self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, self.frame.copy())
        print('saved',filename)

    def showLandmarks(self):
        self.isShowLandmarks = not self.isShowLandmarks

        '''
            if self.isShowLandmarks == True:
            landmark_frame = cv2.resize(self.landmark_frame, None, fx=1 / 4, fy=1 / 4)
            self.frame[0:self.frame.shape[0]*1//4, self.frame.shape[1]*3//4: self.frame.shape[1], :] = landmark_frame

            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            if self.panel is None:
                self.panel = tk.Label(image=image)
                self.panel.image = image
                self.panel.pack(side="left", padx=10, pady=10)

            # otherwise, simply update the panel
            else:
                self.panel.configure(image=image)
                self.panel.image = image
        '''
    def showSource(self):
        self.isShowSource = not self.isShowSource

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        print('events stopping')
        self.cap.release()
        print('cap is releasing')
        self.root.quit()
        print('root has quit')

        #cv2.destroyAllWindows()
        #print('windows being destroyed')
        self.root.destroy()


    def appLoop(self):
        try:

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



            self.imgTarget = cv2.imread("../data/large.jpg")
            if self.isVideoLoop == False:

                #loop through every picture and generate images of myself in other people fom my selfie

                self.imgSource = cv2.imread("../data/AndileIDPhoto.jpg")

                self.source_frame = self.imgSource

                # obtain the texture coordinates of the 2D projection of the 3D model of the target face
                textureCoords = getTextureCoordinates(self.imgTarget, mean3DShape, blendShapes, idxs3D, idxs2D,
                                                      detector,
                                                      predictor)

                # apply this recieved texture coordinates to the image recieved from the webcam
                # generate model on game engine and modify parameters as looping occures
                # this is done so that we need  not load textures from the beginning
                # when rendering the face stolen by the imposter
                faceswapper = FaceRenderer.FaceSwapper(self.imgSource, self.imgTarget, textureCoords, mesh)

                # locate any faces on the frame
                faces = getFacialLandmarks(self.imgSource, detector, predictor, 320)

                # if any faces are detected, this is where we swap them...
                if faces is not None:
                    self.black_image = np.zeros(self.imgSource.shape, np.uint8)
                    for facialLandmarks2D in faces:
                        # draw the landmarks of all the faces detected from the source frame
                        self.black_image = drawImposterLandmarks(self.imgSource, facialLandmarks2D, self.black_image)
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
                        # cv2.imshow('mask', mask)

                        #targetFaceIn3D_WithComplexionImg = FaceProcessor.improvedTransferComlexion(self.imgSource, targetFaceIn3D,
                                                                                           #mask)
                        targetFaceIn3D_WithComplexionImg = FaceProcessor.transferComlexion(self.imgSource, targetFaceIn3D, mask)
                        # cv2.imshow('targetFaceIn3D_WithComplexionImg', targetFaceIn3D_WithComplexionImg)
                        self.imgSource, photoShopped = FaceProcessor.pasteAndBlendNewFace(targetFaceIn3D_WithComplexionImg, self.imgSource, mask)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format


                '''
                imageS = cv2.cvtColor(self.source_frame, cv2.COLOR_BGR2RGB)
                imageS = Image.fromarray(imageS)
                imageS = ImageTk.PhotoImage(imageS)

                imageT = cv2.cvtColor(self.imgTarget, cv2.COLOR_BGR2RGB)
                imageT = Image.fromarray(imageT)
                imageT = ImageTk.PhotoImage(imageT)

                imageR = cv2.cvtColor(self.imgSource, cv2.COLOR_BGR2RGB)
                imageR = Image.fromarray(imageR)
                imageR = ImageTk.PhotoImage(imageR)
                '''

                image = cv2.cvtColor(self.black_image, cv2.COLOR_BGR2RGB)
                image_Resized = cv2.resize(image, self.imageSize)
                image_Resized = Image.fromarray(image_Resized)
                image_Resized = ImageTk.PhotoImage(image_Resized)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    '''
                    if self.isImgSource == True:
                        self.panel = tk.Label(image=imageR)
                        self.panel.image = imageR
                        self.panel.pack(side="top", padx=10, pady=10)
                    elif self.isSource_Frame == True:
                        self.panel = tk.Label(image=imageS)
                        self.panel.image = imageS
                        self.panel.pack(side="top", padx=10, pady=10)
                    elif self.isImgTarget == True:
                        self.panel = tk.Label(image=imageT)
                        self.panel.image = imageT
                        self.panel.pack(side="top", padx=10, pady=10)
                    else:
                    '''

                    self.panel = tk.Label(image=image_Resized)
                    self.panel.image = image_Resized
                    self.panel.pack(side="top", padx=10, pady=10)


                # otherwise, simply update the panel
                else:
                    '''
                    if self.isImgSource == True:
                        self.panel.configure(image=imageR)
                        self.panel.image = imageR
                    elif self.isSource_Frame == True:
                        self.panel.configure(image=imageS)
                        self.panel.image = imageS
                    elif self.isImgTarget == True:
                        self.panel.configure(image=imageT)
                        self.panel.image = imageT
                    else:
                    '''

                    self.panel.configure(image=image_Resized)
                    self.panel.image = image_Resized

                self.frame = self.imgSource.copy()
                '''
                # grab the current timestamp and use it to construct the
                # output path
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
                p = os.path.sep.join((self.outputPath, filename))

                # save the file
                cv2.imwrite(p, self.imgSource.copy())
                '''

            else:
                self.videoLoop()
        except RuntimeError:
            print("error")

    def showVideoOrNot(self):
        self.videoLoop()
        self.isVideoLoop = not self.isVideoLoop

    def showResultImg(self):
        # self.isImgSource = not self.isImgSource
        print(self.imgSource.shape)
        imageR = cv2.cvtColor(self.imgSource, cv2.COLOR_BGR2RGB)
        imageR_Resized = cv2.resize(imageR, self.imageSize)
        imageR_Resized = Image.fromarray(imageR_Resized)
        imageR_Resized = ImageTk.PhotoImage(imageR_Resized)

        if self.panel is None:
            #if self.isImgSource == True:
            self.panel = tk.Label(image=imageR_Resized)
            self.panel.image = imageR_Resized
            self.panel.pack(side="top", padx=10, pady=10)
        else:
            #if self.isImgSource == True:
            self.panel.configure(image=imageR_Resized)
            self.panel.image = imageR_Resized

    def showImposterImg(self):
        # self.isSource_Frame = not self.isSource_Frame

        imageS = cv2.cvtColor(self.source_frame, cv2.COLOR_BGR2RGB)
        imageS_Resized = cv2.resize(imageS, self.imageSize)
        imageS_Resized = Image.fromarray(imageS_Resized)
        imageS_Resized = ImageTk.PhotoImage(imageS_Resized)

        if self.panel is None:
            self.panel = tk.Label(image=imageS_Resized)
            self.panel.image = imageS_Resized
            self.panel.pack(side="top", padx=10, pady=10)
        else:
            #if self.isImgSource == True:
            self.panel.configure(image=imageS_Resized)
            self.panel.image = imageS_Resized

    def showTargetImg(self):
        # self.isImgTarget = not self.isImgTarget

        imageT = cv2.cvtColor(self.imgTarget, cv2.COLOR_BGR2RGB)
        imageT_Resized = cv2.resize(imageT, self.imageSize)
        imageT_Resized = Image.fromarray(imageT_Resized)
        imageT_Resized = ImageTk.PhotoImage(imageT_Resized)

        if self.panel is None:
            self.panel = tk.Label(image=imageT_Resized)
            self.panel.image = imageT_Resized
            self.panel.pack(side="top", padx=10, pady=10)
        else:
            #if self.isImgSource == True:
            self.panel.configure(image=imageT_Resized)
            self.panel.image = imageT_Resized




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

    faceImagePath = "../data/AndileIDPhoto.jpg"

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
            black_image = np.zeros(cameraImage.shape, np.uint8)
            for facialLandmarks2D in faces:
                # draw the landmarks of all the faces detected from the source frame
                onlyLandmarks = drawImposterLandmarks(cameraImage, facialLandmarks2D, black_image)
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
                cv2.imshow('Only Landmarks', onlyLandmarks)
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
    #main()
    #'''
    cap = cv2.VideoCapture(0)
    #time.sleep(5.0)
    os.makedirs('../data/output/imageApp', exist_ok=True)
    outputPath = '../data/output/imageApp'
    ui = UI(cap, outputPath)
    ui.root.mainloop()
    print('---------------')
    #'''
