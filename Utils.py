
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def randomDataGenerator():
    """

    :param theta0_init: slope
    :param theta1_init: bias
    :return: data x,y
    """

    x = np.arange(start=0, stop=5, step=0.01)
    n_rnd = 500
    theta0_init = np.random.normal(loc=1, scale=0.1, size=n_rnd)
    theta1_init = np.random.normal(loc=5, scale=0.2, size=n_rnd)
    y = theta0_init * x + theta1_init
    return x , y, theta0_init, theta1_init


def cost_func_3d(theta0, theta1, dataX, dataY):
    """
    Function to compute the 3D matrix for contour
    """
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((dataY-hypothesis(dataX, theta0, theta1))**2, axis=2)/2

def hypothesis(x, theta0, theta1):
    return theta0*x + theta1

def gradientDescent(m,c,x,y):
    all_m = []
    all_c = []
    i = 1
    threshold = 999999
    while threshold > 10e-20:
        d_m = (np.sum((hypothesis(x,m,c) - y ) * x)) * (2/len(x))
        d_c = (np.sum(hypothesis(x,m,c) - y )) *  (2/len(x))
        old_m = m
        all_m.append(m)
        all_c.append(c)
        m = m - 0.01 * d_m
        c = c - 0.01  * d_c
        threshold = abs(old_m - m)
        i = i +1
        print(m,c, threshold)
    return all_m, all_c



if __name__ == '__main__':

    data = randomDataGenerator()
    print(data[1])
    import matplotlib.pyplot as plt 
    plt.plot(data[0],data[1], '*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Data")
    plt.show()
    print('function was called')