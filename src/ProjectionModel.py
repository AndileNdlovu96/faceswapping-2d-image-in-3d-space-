import numpy as np
import cv2


class ProjectionModel:
    numOfParameters = 6
    n = 0

    def __init__(self, numOfBlendShapes):
        self.n = numOfBlendShapes
        self.numOfParameters += numOfBlendShapes

    def residual(self, X, Y, Beta):
        # the difference between the projected face model and the camera face landmarks
        res = Y - self.f(X, Beta)
        # reduce it to a single vector
        res = res.flatten()
        return res

    def f(self, X, Beta):
        # scale value
        a = Beta[0]

        # rotation values
        # xyz rotation
        r = Beta[1:4]

        # translation values
        # xy translation
        t = Beta[4:6]

        # blendshape weights
        w = Beta[6:]

        # mean3DShape - the neutral face
        S_0 = X[0]

        # the blendshapes - various facial expressions
        S_1_to_n = X[1]

        # rotation matrix from the rotation vector, Rodriguez pattern
        # this makes it possibe to apply rotation to coordinates by
        # matrix multiplication
        r1, _ = cv2.Rodrigues(r)

        # just the x and y rows
        R = r1[:2]

        # allowing for operator broadcasting R1 to R3 by adding new axes
        W_1_to_n = w[:, np.newaxis, np.newaxis]

        # S= S_0 + W_1_to_n*S_1_to_n --> this is the summed faces
        S = S_0 + np.sum((W_1_to_n * S_1_to_n), axis=0)

        # allowing for operator broadcasting R1 to R2 by adding new axis
        T = t[:, np.newaxis]

        # projectedFace = a*R.S + t
        projectedFace = a * np.dot(R, S) + T
        # print(projectedFace.shape)
        return projectedFace

    def J(self, X, Beta):  # there was a y here...
        # scale value
        a = Beta[0]

        # rotation values
        # xyz rotation
        r = Beta[1:4]

        # translation values
        # xy translation
        t = Beta[4:6]

        # blendshape weights
        w = Beta[6:]

        # mean3DShape - the neutral face
        S_0 = X[0]

        # the blendshapes - various facial expressions
        S_i_to_n = X[1]

        # rotation matrix from the rotation vector, Rodriguez pattern
        # this makes it possibe to apply rotation to coordinates by
        # matrix multiplication
        r1, _ = cv2.Rodrigues(r)

        # just the x and y rows
        R = r1[:2]

        # allowing for operator broadcasting R1 to R3 by adding new axes
        W_i_to_n = w[:, np.newaxis, np.newaxis]

        # S= S_0 + W_i_to_n*S_i_to_n --> this is the summed faces
        S = S_0 + np.sum((W_i_to_n * S_i_to_n), axis=0)

        # this is a 3x46 there are 46 points for each x,y,z row of coords
        numOfDataPoints = S.shape[1]

        # nSamples * 2 because every point has two dimensions (x and y)
        j = np.zeros((numOfDataPoints * 2, self.numOfParameters))

        # df/da = R.S
        j[:, 0] = np.dot(R, S).flatten()

        # with this parameter, a principle definition of the a partial derivative will be used
        # df/dr_i = lim of [f(r_i+h)- f(r_i)]/h as h->0 for i={1,2,3}
        h_val = 10e-4
        for i in range(1, 4):
            h = np.zeros(self.numOfParameters)
            h[i] = h_val
            j[:, i] = ((self.f(X, Beta + h) - self.f(X, Beta)) / h_val).flatten()

        # df/dt_i = 1 for i={1, 2}
        # for the x coordinate data points
        j[:numOfDataPoints, 4] = 1
        # for the y coordinate data points
        j[numOfDataPoints:, 5] = 1

        # w_1_to_n Betas cover at J[:, 6:]
        for i in range(6, self.numOfParameters):
            # df/dw_i = a*(R.S_i) + T
            j[:, i] = a * np.dot(R, S_i_to_n[i - 6]).flatten()

        return j

    def getInitialBetas(self, X, Y):
        # center the 3D shape to the origin
        mean3DShape = X.T
        shape3DCentered = mean3DShape - np.mean(mean3DShape, axis=0)

        # center the 2D shape to the origin
        facialLandmarks2D = Y.T
        shape2DCentered = facialLandmarks2D - np.mean(facialLandmarks2D, axis=0)

        # make sure 3D shape is scaled for the 2D shape
        # only consider the xy plane coordinates so that the scale makes sense
        a = np.linalg.norm(shape2DCentered) / np.linalg.norm(shape3DCentered[:, :2])

        # use the difference between coordinates of the facial landmarks and the mean 3D shape
        # in the xy plane to inform the initial translation matrix
        xyDelta = np.mean(facialLandmarks2D, axis=0) - np.mean(mean3DShape[:, :2], axis=0)

        Beta = np.zeros(self.numOfParameters)
        Beta[0] = a
        Beta[4:6] = xyDelta  # [0:2]

        return Beta
