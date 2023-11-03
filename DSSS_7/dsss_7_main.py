from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

##########################################################################

r1 = pd.read_csv("regression_1.csv")
r2 = pd.read_csv("regression_2.csv")
c = pd.read_csv("classification.csv")


##########################################################################

def linearFunction(x, m, b):
    return m * x + b


##########################################################################

def poly_quadratic_function(x, a, b, c, d):
    return a * (x * x * x) + b * (x * x) + c * x + d


##########################################################################

def a(values, x, y):
    # minimize the mean-squared error (MSE)
    return ((y - linearFunction(x, *values)) ** 2).sum()


##########################################################################

def b(values, x, y):
    # minimize the mean-squared error (MSE)
    return ((y - poly_quadratic_function(x, *values)) ** 2).sum()


###########################################################################

x1 = np.asarray(r2['x1'])
y1 = r2['x2']

###########################################################################

x2 = np.asarray(r1['x1'])
y2 = r1['x2']

###########################################################################

m = minimize(a, [1, 1], args=(x1, y1))

###########################################################################

n = minimize(b, [1, 1, 1, 1], args=(x2, y2))

###########################################################################

x = np.arange(-1.0, 8, 0.1)
sin = poly_quadratic_function(x, *n.x)
plt.plot(x, sin, color='green', linewidth=1)
sns.scatterplot(data=r1, x='x1', y='x2', s=100)
plt.show()

###########################################################################

x = np.arange(-7.0, 11.0, 0.1)
parabola_r2 = linearFunction(x, *m.x)
plt.plot(x, parabola_r2, color='green', linewidth=1)
sns.scatterplot(data=r2, x='x1', y='x2', s=100)
plt.show()