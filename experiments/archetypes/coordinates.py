'''
Created on 30 May 2017

@author: alejomc
'''
'''
coordinates.py

2012 Brandon Mechtley
Arizona State University

Various tools for working with barycentric coordinates of arbitrary dimension
including routines for display. All routines automatically normalize 
barycentric coordinates.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

def project_pointline(p, a, b):
    '''
    Euclidean projection of a point onto a line segment.

    Args:
        p (np.ndarray): point of arbitrary dimensionality.
        a (np.ndarray): first endpoint of the line segment.
        b (np.ndarray): second endpoint of the line segment.

    Returns:
        (1, 2) np.ndarray of the projected cartesian coordinates.
    '''

    return a + (np.dot(p - a, b - a) / np.dot(b - a, b - a)) * (b - a)

def bary2cart(bary, corners=None):
    '''
    Convert barycentric coordinates to cartesian coordinates given the
    cartesian coordinates of the corners.

    Args:
        bary (np.ndarray): barycentric coordinates to convert. If this matrix
            has multiple rows, each row is interpreted as an individual
            coordinate to convert.
        corners (np.ndarray): cartesian coordinates of the corners.

    Returns:
        2-column np.ndarray of cartesian coordinates for each barycentric
        coordinate provided.
    '''

    if corners == None:
        corners = polycorners(bary.shape[-1])

    cart = None

    if len(bary.shape) > 1 and bary.shape[1] > 1:
        cart = np.array([np.sum(b / np.sum(b) * corners.T, axis=1) for b in bary])
    else:
        cart = np.sum(bary / np.sum(bary) * corners.T, axis=1)

    return cart

def polycorners(ncorners=3):
    '''
    Return 2D cartesian coordinates of a regular convex polygon of a specified
    number of corners.

    Args:
        ncorners (int, optional) number of corners for the polygon (default 3).

    Returns:
        (ncorners, 2) np.ndarray of cartesian coordinates of the polygon.
    '''

    center = np.array([0.5, 0.5])
    points = []

    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))

    return np.array(points)
# import numpy
# corners = polycorners(3)
# print(corners)
# X = numpy.eye(corners.shape[0])
# print(X)
# cart = bary2cart(X, corners=corners)
# print(cart)


def cart2bary(X, P):
    import numpy
    #numpy.set_printoptions(precision=10)
    #P is points
    #X is the simplex
    M = P.shape[0]
    N = P.shape[1]
    
    assert X.shape[1] == N,  "Simplex X must have same number of columns as point matrix P"
    
    assert X.shape[0] == (N+1), "Simplex X must have N columns and N+1 rows"
    X1 = X[:-1,] - (numpy.multiply(numpy.ones((N,1)), X[-1,]))
    
    assert 1.0/numpy.linalg.cond(X1) > 0.00000001, "Degenerate simplex"

    Beta = numpy.matmul((P - numpy.tile(X[-1,], (M,1))), numpy.linalg.inv(X1))
    
    Beta = numpy.hstack((Beta, (1-numpy.sum(Beta, axis=1).reshape(-1,1))))
    return numpy.round(Beta,10) + 0.0

# print(cart2bary(corners, cart))




def lattice(ncorners=3, sides=False):
    '''
    Create a lattice of linear combinations of barycentric coordinates with 
    ncorners corners. This lattice is constructed from the corners, the center
    point between them, points between the corners and the center, and pairwise
    combinations of the corners and the center.

    Args:
        ncorners (int, optional): number of corners of the boundary polygon 
            (default 3).
        sides (bool, optional): whether or not to include pairwise combinations
            of the corners (i.e. sides) in the lattice (default False).

    Returns:
        np.ndarray of barycentric coordinates for each point in the lattice.
    '''

    # 1. Corners.
    coords = list(np.identity(ncorners))

    # 2. Center.
    center = np.array([1. / ncorners] * ncorners)
    coords.append(center)

    # 3. Corner - Center.
    for i in range(ncorners):
        for j in range(i + 1, ncorners):
            coords.append((coords[i] + coords[j] + center) / 3)

    # 4. Corner - Corner - Center.
    for i in range(ncorners):
        coords.append((coords[i] + center) / 2)

    # 5. Corner - Corner (Sides)
    if sides:
        for i in range(ncorners):
            for j in range(i + 1, ncorners):
                coords.append((coords[i] + coords[j]) / 2)

    # 6. Return unique coordinates (some duplicates using this method with
    # e.g. ncorners=2)
    return np.array(list(set(tuple(c) for c in coords)))

def verttext(pt, txt, center=[.5,.5], dist=1./15, color='red'):
    '''
    Display a text label for a vertex with respect to the center. The text
    will be a certain distance from the specified vertex in the direction of
    the vector extending from the center point.

    Args:
        pt (np.ndarray): two-dimensional array of cartesian coordinates of the
            point to label.
        txt (str): text to display. Text will be horizontally and vertically
            centered around pt.
        center (np.ndarray, optional): reference center point used for 
            arranging the text around pt (default [.5, .5]).
        dist (float, optional): distance between point and the text
            (default 1./15).
        color (str, optional): matplotlib color of the text (default 'red').
    '''

    vert = pt - center
    s = np.sum(np.abs(vert))

    if s == 0:
        vert = np.array([0., 1.])
    else:
        vert /= s

    vert *= dist

    pp.text(
        pt[0] + vert[0],
        pt[1] + vert[1],
        txt,
        horizontalalignment='center',
        verticalalignment='center',
        color=color
    )

def polyshow(
    coords, 
    color=None, 
    label=None, 
    labelvertices=False, 
    polycolor=None, 
    lines=[]
):
    '''
    Plot a regular convex polygon surrounding one or more barycentric 
    coordinates within the it. Vertices and corners will be labeled
    sequentially starting at 0.

    Args:
        coords (np.ndarray or list): one or more barycentric coordinates of 
            equal arbitrary dimension. The dimensionality of the coordinates 
            will correspond to the number of vertices of the polygon that is
            drawn.
        color (str or list, optional): color in which to draw the coords. If
            color is a list of the same length as coords, each entry will 
            correspond to the respective coordinates.
    '''


    # Defaults.
    coords = np.array(coords)
    if len(coords.shape) < 2: coords = [coords]
    for coord in coords:
        if np.sum(coord) > 0:
            coord /= np.sum(coord)

    if color == None: color = 'blue'
    if type(color) == str: color = [color] * len(coords)

    if label == None: label = ''
    if type(label) == str: label = [label] * len(coords)

    # Number of sides.
    d = len(coords[0])

    # Cartesian coordinates of the vertices of the polygon and each point.
    corners = polycorners(d)
    cart = np.array([
        np.sum([
            c * cnr 
            for c, cnr in zip(coord, corners)
        ], axis=0) 
        for coord in coords
    ])

    # Figure/axes setup.
    f = pp.figure(frameon=False)
    ax = pp.axes(frameon=False)
    ax.axis('equal')
    pp.xticks([])
    pp.yticks([])

    # Add the polygon and its vertices to the figure.
    ax.add_patch(pp.Polygon(corners, closed=True, fill=False, alpha=0.5))
    ax.scatter(corners[:,0], corners[:,1], color='red', s=50)
    if labelvertices:
        map(lambda i: verttext(corners[i], '$v_{%d}$' % i), range(len(corners)))

    # Add any extra lines to the figure.
    map(ax.add_line, lines)

    # Add the interior points and their labels.
    ax.scatter(cart[:,0], cart[:,1], color=color, s=100)

    for c, txt, clr in zip(cart, label, color):
        verttext(c, txt, color=clr)

    return f

def baryedges(coords, sidecoords=None):
    '''
    Return an array of barycentric coordinates corresponding to the closest
    point on each edge of the respective polygon.

    Args:
        coords (np.ndarray): input coordinates.

    Returns:
        np.ndarray of barycentric coordinates.
    '''

    # There ought to be an easier way to do this without switching out of
    # barycentric coordinates, but after obsessing over it for an entire day,
    # I'll have to work on that later.

    d = len(coords)                     # number of edges.
    corners = polycorners(d)            # cart. corners of the polygon.
    cart = bary2cart(coords, corners)   # cart. coordinates of the input point.

    e = np.zeros((d, d))                # bary. coords. for proj. to each side.

    # Pairs of corners that form sides (01, 12, 20 for a triange).
    pairs = zip(np.arange(d), np.roll(np.arange(d), -1))

    # Proj. the pt onto each edge in cart. space, put it back into bary. coords.
    for i1, i2 in pairs:
        proj = project_pointline(cart, corners[i1], corners[i2])
        
        distances = np.array([
            np.linalg.norm(corners[a] - proj, 2) 
            for a in [i1, i2]
        ])

        sidebary = 1.0 - distances / np.sum(distances)
        e[i1, (i1, i2)] = sidebary

    if sidecoords:
        return np.array([e[i1, (i1, i2)] for i1, i2 in pairs])

    return e

def circumcircle(a, b, c):
    '''
    Return the center coordinates of a circle that circumscribes an input
    triangle defined by 2D vertices a, b, and c.

    Args:
        a ((number, number) or np.ndarray): first vertex of the input triangle.
        b ((number, number) or np.ndarray): second vertex of the input triangle.
        c ((number, number) or np.ndarray): third vertex of the input triangle.

    Returns:
        Center coordinates of circumscribing circle in form (x, y).
    '''

    ax, ay = a
    bx, by = b
    cx, cy = c
    ax2, ay2, bx2, by2, cx2, cy2 = [d ** 2 for d in [ax, ay, bx, by, cx, cy]]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    ux = (
        (ax2 + ay2) * (by - cy) +
        (bx2 + by2) * (cy - ay) +
        (cx2 + cy2) * (ay - by)
    ) / d

    uy = (
        (ax2 + ay2) * (cx - bx) +
        (bx2 + by2) * (ax - cx) +
        (cx2 + cy2) * (bx - ax)
    ) / d

    return ux, uy

def voronoi(x, y):
    '''
    Return a list of voronoi cells for a collection of points.

    Args:
        x (list or np.ndarray): list of coordinates' x components.
        y (list or np.ndarray): list of coordinates' y components.

    Returns:
        (cells, triangles), where cells is a list of voronoi cells, each once
        containing a list of two-dimensional points; and triangles is a list of
        the triangles from a Delaunay triangulation.
    '''

    p = np.array([a for a in zip(x, y)])
    d = mpl.tri.Triangulation(x, y)
    t = d.triangles
    n = t.shape[0]
    
    # Get circle for each triangle, center will be a voronoi cell point.
    cells = [[] for i in range(x.size)]

    for i in range(n):
        v = [p[t[i,j]] for j in range(3)]
        pt = circumcircle(v[0], v[1], v[2])

        cells[t[i,0]].append(pt)
        cells[t[i,1]].append(pt)
        cells[t[i,2]].append(pt)

    # Reordering cell p in trigonometric way
    for i, cell in enumerate(cells):
        xy = np.array(cell)
        order = np.argsort(np.arctan2(xy[:,1] - y[i], xy[:,0] - x[i]))

        cell = xy[order].tolist()
        cell.append(cell[0])

        cells[i] = cell

    return cells


def baryplot(
    values, 
    points=None, 
    labels='abc', 
    cmap=mpl.cm.BrBG, 
    clabel='', 
    vmin=None, 
    vmax=None
):
    '''
    Create a triangular voronoi cell pseudocolor plot. Create a voronoi diagram
    for each coordinate (points) within the triangle and color each cell
    according to its value (values).

    Args:
        values (list or np.ndarray): list of scalar values for each barycentric 
            point.
        points (list or np.ndarray): list of three-parameter barycentric 
            points.
        labels (list): list of three label strings, one for each side of the
            triangle.
        cmap (matplotlib.colors.Colormap): colormap for pseudocolor plot.
        clabel (str): colorbar label for values
        vmin (number): minimum value for coloring. If None, use minimum of the
            input values.
        vmax (number): maximum value for coloring. If None, use maximum of the
            input values.
    '''

    if points is None: points = []

    vmin = vmin if vmin is not None else np.amin(values)
    vmax = vmax if vmax is not None else np.amax(values)

    p = bary2cart(points) if len(points) else bary2cart(lattice(3))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(values)

    #values = (values - np.amin(values)) / (np.amax(values) - np.amin(values))
    cells = voronoi(p[:,0], p[:,1])

    xmin, xmax, xavg = np.amin(p[:,0]), np.amax(p[:,0]), np.mean(p[:,0])
    ymin, ymax, yavg = np.amin(p[:,1]), np.amax(p[:,1]), np.mean(p[:,1])

    s60, c60 = np.sin(np.pi / 3.), np.cos(np.pi / 3.)
    s30, c30 = np.sin(np.pi / 6.), np.cos(np.pi / 6.)

    # Start drawing.
    ax = pp.gca()

    # Clipping triangle for the voronoi patches.
    clip = mpl.patches.Polygon([
        (xmin, ymin), (xmax, ymin), (xavg, ymax),
    ], transform=ax.transData)

    # Draw voronoi patches.
    for i, cell in enumerate(cells):
        codes = [mpl.path.Path.MOVETO] \
            + [mpl.path.Path.LINETO] * (len(cell) - 2) \
            + [mpl.path.Path.CLOSEPOLY]

        pth = mpl.path.Path(cell, codes)

        patch = mpl.patches.PathPatch(
            pth,
            zorder=-1,
            facecolor=colors[i],
            clip_path=clip,
            edgecolor='none'
        )

        ax.add_patch(patch)

    # Add barycentric labels for vertices.
    ax.text(xmin - .0125, ymin - .02, '$(0,1,0)$', ha='right', va='center')
    ax.text(xmax + .0125, ymin - .02, '$(0,0,1)$', ha='left', va='center')
    ax.text(xavg, ymax + .035, '$(1,0,0)$', ha='center', va='bottom')

    # Labels.
    ax.text(
        xavg + c30 * .35, yavg + s30 * .35,
        labels[2], ha='center', va='center', rotation=-60
    )

    ax.text(
        xavg, ymin - .05,
        labels[1], ha='center', va='top'
    )

    ax.text(
        xavg - c30 * .35, yavg + s30 * .35,
        labels[0], ha='center', va='center', rotation=60
    )

    arrowopts = dict(
        width=.00125,
        frac=.0125,
        headwidth=.01,
        transform=ax.transData
    )

    fig = pp.gcf()

    # Arrows along edges.
    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xmin - c60 * .025, ymin + s60 * .025),
        (xavg - c60 * .025, ymax + s60 * .025),
        **arrowopts
    ))

    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xmax, ymin - .025),
        (xmin, ymin - .025),
        **arrowopts
    ))

    ax.add_patch(mpl.patches.YAArrow(
        fig,
        (xavg + c60 * .025, ymax + s60 * .025),
        (xmax + c60 * .025, ymin + s60 * .025),
        **arrowopts
    ))

    # Make axes equal, get rid of border.
    pp.axis('equal')
    ax.axis([
        xmin - c60 * .2, xmax + c60 * .2,
        ymin - s60 * .2, ymax + s60 * .2
    ])
    pp.axis('off')

    cax, kw = mpl.colorbar.make_axes(ax, orientation='vertical', shrink=0.7)
    cb = mpl.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
        orientation='vertical',
        ticks=np.linspace(vmin, vmax, 5)
    )

    cb.set_label(clabel)