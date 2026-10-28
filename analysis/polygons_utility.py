from .standard_variables import ellipse_params_labels

import numpy as np
import shapely.geometry.polygon as shp
import time

def ellipse(*args, N=128):
    theta = np.linspace(0, 2 * np.pi, N)[np.newaxis, :]
    xc, yc, a, b, phi = args
    x = xc + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi)
    y = yc + a * np.cos(theta) * np.sin(phi) + b * np.sin(theta) * np.cos(phi)
    return x, y


def reshape_coord_for_poly(x, y):
    """

    Parameters
    ----------
    x : 1D numpy array or list
        x position of the polygon points
    y : 1D numpy array or list
        y position of the polygon points

    Returns
    -------
    coords : 2D numpy array
        Readable list coordinates for shapely module

    """
    coords = np.reshape(np.ravel([x, y]), (len(x), 2), order='F')
    return coords

def reshape_coord_for_poly_vec(x, y):
    #coords = np.reshape(np.ravel([x, y]), (x.shape[0], 2, x.shape[1]), order='F')
    coords = np.dstack((x, y))
    return coords

def buildPolygons(catalog, obj_type, N=128):
    from .load import ObjectType
    
    if obj_type in (ObjectType.ELLIPSE, ObjectType.YSO):
        args = np.array([catalog[label].to_numpy()[:, np.newaxis] for label in ellipse_params_labels])
        x, y = ellipse(*args, N=N)
        coords = reshape_coord_for_poly_vec(x, y)

    elif obj_type == ObjectType.POLYGON:
        x, y = catalog[ellipse_params_labels[0]], catalog[ellipse_params_labels[1]]
        coords = []
        for xx, yy in zip(x, y):
            xx = np.asarray(xx, dtype=float)
            yy = np.asarray(yy, dtype=float)
            coords.append(np.dstack((xx, yy))[0])

    """
    old script for polygons instead of ellipses parameters
    x, y = catalog[strings[0]], catalog[strings[1]]
    coords = []
    for xx, yy in zip(x, y):
        coords.append(reshape_coord_for_poly(xx, yy))
        #print(len(catalog), xx, yy)
        #pp = shp.Polygon(reshape_coord_for_poly(xx, yy))
        #if not pp.is_valid:
        #    print(len(catalog), xx, yy)
    """
    return [shp.Polygon(c) for c in coords]

def overlapMatrix(polygons1, polygons2):
    """
    Returns the included matrix between 2 sets of shapely polygons (boolean), and their %area
    """
    # polygons1, larger scale
    # polygons2, smaller scale
    from shapely.strtree import STRtree

    area = np.zeros(shape=(len(polygons1), len(polygons2)))
    tree = STRtree(polygons2)

    arr_indices = tree.query(polygons1, predicate="intersects")    
    for idx1, idx2 in zip(*arr_indices):
        if polygons2[idx2].within(polygons1[idx1]):
            area[idx1, idx2] = 1
        else:
            intersct = polygons2[idx2].intersection(polygons1[idx1])
            area[idx1, idx2] = intersct.area / min(polygons1[idx1].area, polygons2[idx2].area)    
    return area