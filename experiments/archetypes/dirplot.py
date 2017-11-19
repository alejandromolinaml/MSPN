'''
Created on 29 May 2017

@author: alejomc
'''
from _functools import reduce

from dirichlet.simplex import barycentric, cartesian
import numpy

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


#from http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
corners = np.array([cartesian((1,0,0)), cartesian((0,1,0)), cartesian((0,0,1))])

triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# plt.figure(figsize=(8, 4))
# for (i, mesh) in enumerate((triangle, trimesh)):
#     plt.subplot(1, 2, i+ 1)
#     plt.triplot(mesh)
#     plt.axis('off')
#     plt.axis('equal')
    


class Dirichlet2plot(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])
        
        
def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math
    tol=1.e-3
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    #print(trimesh.x, trimesh.y)
    pvals = [dist.pdf(np.clip(barycentric(xy), tol, 1.0 - tol)) for xy in zip(trimesh.x, trimesh.y)]
    #print(pvals)
    
    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    
def draw_pdf_contours_func(func, nlevels=200, subdiv=9, **kwargs):
    import math
    tol=1.e-6
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    
    
    pvals = [func(np.clip(barycentric(xy), tol, 1.0 - tol)) for xy in zip(trimesh.x, trimesh.y)]


    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    
def draw_pdf_contours_func2(func, nlevels=30, subdiv=2, **kwargs):
    import math
    tol=1.e-3
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [func(np.clip(barycentric(xy), tol, 1.0 - tol)) for xy in zip(trimesh.x, trimesh.y)]

    bounds=np.linspace(-2,14,40)                    
    tf = plt.tricontourf(trimesh, pvals, nlevels, levels = bounds, extend = 'both', **kwargs)
    plt.colorbar(tf) 
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    
#     
# def plot3ddensity(data, func):
#     from scipy.interpolate import griddata
#     import numpy as np
#     
#     # Create some test data, 3D gaussian, 200 points
#     dx, pts = 2, 100j
#     
#     N = 500
#     R = np.random.random((N,3))*2*dx - dx
#     V = np.exp(-( (R**2).sum(axis=1)) )
#     
#     # Create the grid to interpolate on
#     X,Y,Z = np.mgrid[-dx:dx:pts, -dx:dx:pts, -dx:dx:pts]
#     
#     # Interpolate the data
#     F = griddata(R, V, (X,Y,Z))
#     
#     from mayavi.mlab import *
#     contour3d(F,contours=8,opacity=.2 )