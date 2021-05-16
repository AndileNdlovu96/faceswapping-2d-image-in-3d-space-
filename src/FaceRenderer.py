import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

VV_Distance = 10e3

def setOrthographicProjection(dim):
    (w, h) = dim
    # begin on the projection stack
    # this stack handles how face is projected/percieved
    # and what is clipped out

    # begin by finding the current projection matrix stack
    glMatrixMode(GL_PROJECTION)
    # clear the stack
    # ensure that initial projection matrix is just the identity matrix
    # transformations and other commands are mostly matrix multiplication
    glLoadIdentity()
    # we are using orthographic projection
    # we needn't consider how far away our texture image's 3D model is from camera
    # i.e. we want to keep the relative size of translated face as is
    glOrtho(0, w, h, 0, -1*VV_Distance, VV_Distance)

    # returning to the model-view stack
    glMatrixMode(GL_MODELVIEW)

# LOAD DATA INTO 2D TEXTURE
def addTexture(img):
    # generate an unused name for the face texture object.
    # we want only 1 unused texture name
    textureName = glGenTextures(1)

    # if this texture name is not the reserved '0'
    # or isn't a nonzero name used by an existing texture object
    #if glIsTexture == GL_FALSE: <-- this $hit doesn't behave as I expected...

    # create a texture object to be identified by its texture name
    # with the assumed target being 2-D and set to default state values of 2d texture objects
    glBindTexture(GL_TEXTURE_2D, textureName)

    # specifies how data is unpacked from application to memory
    # affects the efficiency of writes of pixel data to OpenGL memory (graphics driver?)
    # memory alignment affects CPU read/wrote speeds
    # alignment must be multiple of word size (in this case 32-bit / 4bytes)
    # no padding between each pixel data row. --> (1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # specifies how we allocate storage for the texture and pass data to texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, img)

    # for floating point parameters - they seem to work faster than integers
    # we want to modify the 2D texture unit's mag filter value to nearest
    # mag filter is the function used whenever pixel maps to area <= 1 texture element
    # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
    # nearest is apparently faster
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # we want to modify the 2D texture unit's min filter value to nearest
    # min filter is the function used whenever pixel maps to area > 1 texture element
    # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    # this is to specify how the texture should be placed on surface
    # will put the texture in the surface without bothering about the material of the surface
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    return textureName

class FaceSwapper:

    def __init__(self, cameraImg, textureImg, textureCoords, mesh):
        self.cameraDim = (cameraImg.shape[1], cameraImg.shape[0])
        self.textureDim = (textureImg.shape[1], textureImg.shape[0])

        # initiate pygame in it's own window
        pygame.init()

        # 'DOUBLEBUF' for monitor's refresh rate | 'OPENGL' so pygame knows
        # that opengl is being used
        pygame.display.set_mode(self.cameraDim, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Target Face 3D Model")
        # ORTHOGRAPHIC PROJECTION
        # set up the orthographic viewing volume
        #(w, h) = self.cameraDim
        # begin on the projection stack
        # this stack handles how face is projected/percieved
        # and what is clipped out

        # begin by finding the current projection matrix stack
        glMatrixMode(GL_PROJECTION)
        # clear the stack
        # ensure that initial projection matrix is just the identity matrix
        # transformations and other commands are mostly matrix multiplication
        glLoadIdentity()
        # we are using orthographic projection
        # we needn't consider how far away our texture image's 3D model is from camera
        # i.e. we want to keep the relative size of translated face as is
        glOrtho(0, self.cameraDim[0], self.cameraDim[1], 0, -1 * VV_Distance, VV_Distance)

        # returning to the model-view stack
        glMatrixMode(GL_MODELVIEW)

        # hidden surface removal
        # render closer fragments and not ones further away from viewport/near plane
        glEnable(GL_DEPTH_TEST)

        # enable texture mapping from texture image to 3D model
        glEnable(GL_TEXTURE_2D)

        self.textureCoords = textureCoords
        # normalize texture coordinates
        # this ensures that they are invarient to scale of texture image
        # saves time and computation...
        # all x coordinates divided by texture image width
        self.textureCoords[0, :] /= self.textureDim[0]
        # all y coordinates divided by texture image height
        self.textureCoords[1, :] /= self.textureDim[1]

        self.mesh = mesh
        '''
        this was for testing purposes
        # -------------------GENERATE TEXTURE FOR ORIGINAL FACE--------------------
        # generate an unused name for the face texture object.
        # we want only 1 unused texture name
        self.originalFaceTexture = glGenTextures(1)

        # if this texture name is not the reserved '0'
        # or isn't a nonzero name used by an existing texture object
        # if glIsTexture == GL_FALSE: <-- this $hit doesn't behave as I expected...

        # create a texture object to be identified by its texture name
        # with the assumed target being 2-D and set to default state values of 2d texture objects
        glBindTexture(GL_TEXTURE_2D, self.originalFaceTexture)

        # specifies how data is unpacked from application to memory
        # affects the efficiency of writes of pixel data to OpenGL memory (graphics driver?)
        # memory alignment affects CPU read/wrote speeds
        # alignment must be multiple of word size (in this case 32-bit / 4bytes)
        # no padding between each pixel data row. --> (1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # specifies how we allocate storage for the texture and pass data to texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.cameraDim[0], self.cameraDim[1], 0, GL_BGR, GL_UNSIGNED_BYTE,
                     cameraImg)

        # for floating point parameters - they seem to work faster than integers
        # we want to modify the 2D texture unit's mag filter value to nearest
        # mag filter is the function used whenever pixel maps to area <= 1 texture element
        # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
        # nearest is apparently faster
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # we want to modify the 2D texture unit's min filter value to nearest
        # min filter is the function used whenever pixel maps to area > 1 texture element
        # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # this is to specify how the texture should be placed on surface
        # will put the texture in the surface without bothering about the material of the surface
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        '''




        # -------------------GENERATE TEXTURE FOR NEW FACE--------------------
        # generate an unused name for the face texture object.
        # we want only 1 unused texture name
        self.newFaceTexture = glGenTextures(1)

        # if this texture name is not the reserved '0'
        # or isn't a nonzero name used by an existing texture object
        # if glIsTexture == GL_FALSE: <-- this $hit doesn't behave as I expected...

        # create a texture object to be identified by its texture name
        # with the assumed target being 2-D and set to default state values of 2d texture objects
        glBindTexture(GL_TEXTURE_2D, self.newFaceTexture)

        # specifies how data is unpacked from application to memory
        # affects the efficiency of writes of pixel data to OpenGL memory (graphics driver?)
        # memory alignment affects CPU read/wrote speeds
        # alignment must be multiple of word size (in this case 32-bit / 4bytes)
        # no padding between each pixel data row. --> (1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # specifies how we allocate storage for the texture and pass data to texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.textureDim[0], self.textureDim[1], 0, GL_BGR, GL_UNSIGNED_BYTE,
                     textureImg)

        # for floating point parameters - they seem to work faster than integers
        # we want to modify the 2D texture unit's mag filter value to nearest
        # mag filter is the function used whenever pixel maps to area <= 1 texture element
        # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
        # nearest is apparently faster
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # we want to modify the 2D texture unit's min filter value to nearest
        # min filter is the function used whenever pixel maps to area > 1 texture element
        # GL_NEAREST is the manhattan distance between the texture element and the center of textured pixel
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # this is to specify how the texture should be placed on surface
        # will put the texture in the surface without bothering about the material of the surface
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)



    def renderFace(self, projected3DFace):
        # buffers must be cleared before each iteration of model rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # bind new face texture object to current context
        glBindTexture(GL_TEXTURE_2D, self.newFaceTexture)

        # begin drawing the model with texture overlayed
        glBegin(GL_TRIANGLES)
        for meshIndices in self.mesh:
            for vertex in meshIndices:
                # set desired texture coordinates
                glTexCoord2fv(self.textureCoords[:, vertex])
                # set the desired position of the texel on the viewpoint
                glVertex3fv(projected3DFace[:, vertex])
        glEnd()

        # x, y ->(0,0) specifies the window coordinates of the first pixel that is read from the frame buffer.
        # This location is the lower left corner of a rectangular block of pixels.

        # width, height specify the dimensions of the pixel rectangle.width and height of one correspond to
        # a single pixel.

        data = glReadPixels(0, 0, self.cameraDim[0], self.cameraDim[1], GL_BGR, GL_UNSIGNED_BYTE)
        #print(data)
        renderedImg = np.fromstring(data, dtype=np.uint8)
        #print(renderedImg.shape)
        renderedImg = renderedImg.reshape((self.cameraDim[1], self.cameraDim[0], 3))
        #print('renderedImg.shape', renderedImg.shape)
        # for every BGR value channel of the rendered image...
        channel = renderedImg.shape[2]
        for i in range(channel):
            # computer programs render grids as origion being top left corner,
            # not bottom left like in normal cartesian plane
            renderedImg[:, :, i] = np.flipud(renderedImg[:, :, i])

        # update screen with what has been drawn
        pygame.display.flip()
        # a failed attempt to try and minimize the pygame window...
        #pygame.display.iconify()
        return renderedImg