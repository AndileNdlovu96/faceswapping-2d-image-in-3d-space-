import numpy as np
from scipy import optimize

# Gauss-Newton method will work since in the nature of the problem,
# we have more 46 xy coords and only 20 parameters in total
def GNA(initialBeta, f, J, X, Y, maxIter=10, epsilon=10e-7, showPerformance=True):
    # allow for refined decial values in Beta(the model parameters)
    Beta = np.array(initialBeta, dtype=np.float64)
    #print('B', Beta)
    prevCost=-1

    for i in range(0, maxIter):
        # we are minimizing the residual ie the difference
        # between the data points we have and what we are fitting for
        # -->in this case we are fitting for the facial landmarks given
        #    through the camera using the constructed 3D model of the
        #    face in the texture image
        res = f(X, Y, Beta)
        #print('res', res)

        # the square difference between model and datapoints is called cost
        currCost = np.linalg.norm(res) ** 2
        #print("currCost", currCost)
        if showPerformance:
            print("Cost at iteration", i, ":    ", currCost)

        # there is no need to optimize if the algorithm has converged
        # hence we must check for this in order to further optimize the search
        if algorithmConverges(currCost, prevCost, epsilon):
            break
        # recording the cost obtained from this iteration
        prevCost = currCost

        # we will optimize via the Gauss-Newton Method
        # to obtain the direction of descent
        y = getDirectionOfDescent(Beta, res, J, X)

        # we will further optimize Gauss-Newton Method to safe-guard against
        # divergence by dampening the descent direction
        alpha = getStepSize(LineSearchFun, Beta, y, f, X, Y)

        Beta = Beta + alpha * y

    if showPerformance:
        print("Gauss Newton finished after", i + 1, "iterations")
        res = f(X, Y, Beta)
        cost = np.sum(res ** 2)
        print("cost = ", currCost)
        print("Beta = ", Beta)

    return Beta

def SD_HillClimbing(initialBeta, f, J, X, Y, maxIter=10, epsilon=10e-7, showPerformance=True):
    # allow for refined decial values in Beta(the model parameters)
    Beta = np.array(initialBeta, dtype=np.float64)

    prevCost = -1

    for i in range(0, maxIter):
        # we are minimizing the residual ie the difference
        # between the data points we have and what we are fitting for
        # -->in this case we are fitting for the facial landmarks given
        #    through the camera using the constructed 3D model of the
        #    face in the texture image
        res = f(X, Y, Beta)
        # the square difference between model and datapoints is called cost
        currCost = np.linalg.norm(res) ** 2
        if showPerformance:
            print("Cost at iteration", i, ":    ", currCost)

        # there is no need to optimize if the algorithm has converged
        # hence we must check for this in order to further optimize the search
        if algorithmConverges(currCost, prevCost, epsilon):
            break

        prevCost = currCost

        # establish the Jacobian of the function being minimized
        J_f = J(X, Beta)

        # establish the gradient of the function being minimized
        # g= J_f^T.r
        g = np.dot(J_f.T, res)

        # further optimize Steepest descent Hill Climbing Method to safe-guard against
        # divergence by dampening the descent direction
        lineSearchRes = optimize.minimize_scalar(LineSearchFun, args=(Beta, g, f, X, Y))

        alpha = lineSearchRes.x#["Beta"]

        Beta = Beta + alpha * g

    if showPerformance:
        print("Steepest Descent finished after", i + 1, "iterations")
        res = f(X, Y, Beta)
        cost = np.sum(res ** 2)
        print("cost = ", currCost)
        print("Beta = ", Beta)

    return Beta

def algorithmConverges(currCost, prevCost, epsilon):
    delta = abs(currCost - prevCost)
    # if the cost is hardly changing
    if delta < epsilon:
        return True
    # or if the cost is already good enough
    elif currCost < epsilon:
        return True
    else:
        return False

def LineSearchFun(alpha, Beta, y, f, X, Y):
    # the only varying parameter - which is the parameter subject to our optimization -
    # is Beta and X, Y, and y are kept constant
    adjustedBeta = Beta + alpha * y
    res = f(X, Y, adjustedBeta)
    # the result of this function at each varying alpha value
    # is what will be kept by sciPy's optimize.minimize_scalar
    # function and the alpha that yields the least value is going
    # to be considered as the minimizing step-size
    return np.sum(res**2)

def getStepSize(LineSearchFun, Beta, y, f, X, Y):
    return optimize.minimize_scalar(LineSearchFun, args=(Beta, y, f, X, Y)).x

    #return lineSearchRes.x  # ["x"]

def getDirectionOfDescent(Beta, res,J ,X):
    # establish the Jacobian of the function being minimized
    J_f = J(X, Beta)
    # print('J_f.shape',J_f.shape)
    # establish the gradient of the function being minimized
    # g= J_f^T.r
    g = np.dot(J_f.T, res)
    # print('res', res.shape)
    # print('g', g)
    # establish the Hessian of the function being minimized
    # H = J_f^T.J_f
    H = np.dot(J_f.T, J_f)
    # print('H', H)
    # Now, it would be enough to just compute H^(-1).g
    # however,
    # i) this may not work if H is not PSD
    # ii) it is more computationally efficient for larger matrices
    #     to rather solve for y such that Hy=g instead of computing H^(-1)
    #     NB: the condition number of the function could be checked to see this,
    #         however, it also requires the computation of H^(-1).
    y = np.linalg.solve(H, g)
    return y