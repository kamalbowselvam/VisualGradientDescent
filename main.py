from Utils import randomDataGenerator, gradientDescent, cost_func_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np


def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation

def animate(i):
    line.set_data(all_m[:i],all_c[:i])
    annotation.set_text('Cost = %.2f e10' % (all_c[i] / 10000000000))
    return line, annotation



def init_line():
    line2.set_data([], [])
    annotation.set_text('')
    return line2, annotation

def animate_line(i):
    x = np.linspace(-5, 10, 1000)
    y = all_m[i] * x + all_c[i]
    print(all_m[i],all_c[i])
    line2.set_data(x,y)
    annotation.set_text('Cost = %.2f e10' % (all_c[i] / 10000000000))
    return line2, annotation


if __name__ == "__main__":

    dataX, dataY, theta0_init, theta1_init = randomDataGenerator()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    theta0_grid = np.linspace(-10, 15, 101)
    theta1_grid = np.linspace(-10, 15, 101)
    Z = cost_func_3d(theta0_grid[np.newaxis, :, np.newaxis],
                     theta1_grid[:, np.newaxis, np.newaxis],
                     dataX,
                     dataY)

    X, Y = np.meshgrid(theta0_grid, theta1_grid)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

    all_m, all_c = gradientDescent(m=-10, c=10, x=dataX, y=dataY)



    fig = plt.figure()
    ax = plt.axes()
    ax.contour(X, Y, Z, 100)
    plt.title('Contour Plot for Grandient Descent Convergence')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.scatter(0.9992458979704978, 5.015957571468284, color='blue')
    line, = ax.plot([], [],'r-', lw=2)
    annotation = ax.text(-1, 700000, '')
    annotation.set_animated(True)
    plt.close()

    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=0, blit=True)
    anim.save('animation2.gif', writer='imagemagick', fps=30)
    
    """

    fig = plt.figure()
    ax = plt.axes()
    plt.title('Contour Plot for Grandient Descent Convergence')
    plt.xlabel('x')
    plt.ylabel('y')


    plt.scatter(dataX, dataY, color='blue')
    line2, = ax.plot([], [], 'r-', lw=2)
    annotation = ax.text(-1, 700000, '')
    annotation.set_animated(True)
    plt.close()

    anim = animation.FuncAnimation(fig, animate_line, init_func=init_line, frames=5000, interval=500, blit=True)
    anim.save('animation_plot.gif', writer='imagemagick', fps=30)

    """
