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

def buildPolygons(catalog, strings, N=128, ptype = 1):
    if ptype != 2:
        args = np.array([catalog[s].to_numpy()[:, np.newaxis] for s in strings])
        x, y = ellipse(*args, N=N)
        #print(x)
        coords = reshape_coord_for_poly_vec(x, y)
        #print(coords.shape)
    else:
        x, y = catalog[strings[0]], catalog[strings[1]]
        coords = []
        for xx, yy in zip(x, y):
            coords.append(reshape_coord_for_poly(xx, yy))
            #print(len(catalog), xx, yy)
            #pp = shp.Polygon(reshape_coord_for_poly(xx, yy))
            #if not pp.is_valid:
            #    print(len(catalog), xx, yy)

    return [shp.Polygon(c) for c in coords]

def testMatrix(mat, size, p):
    """
    Matrice booléenne où la condition est vraie entre les tailles (size) et les distances centroids (mat)
    """
    return size / mat > p

def sizeMatrix(polygons):
    """
    Calcule la matrice des tailles moyennes des polygons
    """
    n = len(polygons)
    mat = np.zeros(shape=(n, n))
    for i, p1 in enumerate(polygons):
        for j, p2 in enumerate(polygons):
            if j > i:
                mat[i, j] = np.sqrt(p1.area) + np.sqrt(p2.area)
    mat += np.transpose(mat)
    return mat


def coordMatrix(polygons):
    """
    Calcule les listes des coordonnées des polygons
    """
    n = len(polygons)
    lst = np.zeros(shape=(n, 2))
    lst[:, 0] = [p.centroid.xy[0][0] for i, p in enumerate(polygons)]
    lst[:, 1] = [p.centroid.xy[1][0] for i, p in enumerate(polygons)]
    return lst


def sepAngular(lst):
    from PyAstronomy import pyasl
    mat = pyasl.getAngDist(lst[:, 0, np.newaxis], lst[:, 1, np.newaxis],
                           lst[np.newaxis, :, 0], lst[np.newaxis, :, 1])
    return mat

def sepDistance(lst):
    """
    Calcule la matrice des distances entre les centroids
    """
    mat = np.sqrt(
        (lst[:, 0, np.newaxis] - lst[np.newaxis, :, 0]) ** 2 + (lst[:, 1, np.newaxis] - lst[np.newaxis, :, 1]) ** 2)
    return mat

def minDistance(polygons, mat, replace, p=0.05, verbose=False):
    from PyAstronomy import pyasl
    from shapely.ops import nearest_points
    from tqdm import tqdm

    """
    A partir de la matrice booléenne calcule les termes min distance polygon et les remplace dans mat
    """
    replace = np.triu(replace, k=0)
    idx = np.where(replace)

    if verbose:
        print("Processing minimal distances")
        time.sleep(0.5)

    for i, x in enumerate(tqdm(idx[0], disable=not verbose)):
        # d = polygons[x].distance(polygons[idx[1][i]])
        p1, p2 = nearest_points(polygons[x], polygons[idx[1][i]])
        d = pyasl.getAngDist(p1.x, p1.y, p2.x, p2.y)
        mat[x, idx[1][i]] = d
        mat[idx[1][i], x] = d
    return mat


def distancePolyst(polygons, p=0.05, verbose=False):
    """
    calcul la distance entre les polygons avec la tolérance p (rapide)
    """
    lst = coordMatrix(polygons)
    mat = sepAngular(lst)
    size = sizeMatrix(polygons)
    replace = testMatrix(mat, size, p)

    if verbose:
        print("There are", np.sum(replace) // 2, "paire(s) of polygons that needs minimal distance computation.")
        time.sleep(0.5)

    return minDistance(polygons, mat, replace, p, verbose)


def mindistancePolyst(polygons, verbose=False):
    """
    calcul la distance min entre les polygons
    """
    from PyAstronomy import pyasl
    from shapely.ops import nearest_points
    from tqdm import tqdm

    n = len(polygons)
    mat = np.zeros(shape=(n, n))

    for i in tqdm(range(n), disable=not verbose):
        for j in range(i):
            p1, p2 = nearest_points(polygons[i], polygons[j])
            d = pyasl.getAngDist(p1.x, p1.y, p2.x, p2.y)
            mat[i, j] = d
            mat[j, i] = d

    return mat

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