import linear
from imports import *
import datasets
import math
import pylab
import numpy

x, trajectory = gd.gd(lambda x: x**3+x**10-x**2, lambda x: 3*(x**2)+10*(x**9)-2*x, -1, 100, 0.01)
plot(trajectory)
pylab.plot()
xlabel('Iteration Number')
ylabel('Approximated Value')
ylim([0,50])
suptitle('Plot of x^3+x^10-x^2')
show(True)
