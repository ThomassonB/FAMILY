#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 08:55:47 2021

@author: thomaben
"""
import numpy as np
from astropy.io import fits


def OpenImage(image, path=True):
    """

    Parameters
    ----------
    path
    image : str or HDU
        Can be a path to a .fits image or an already opened .fits image

    Returns
    -------
    img : 2D numpy array
        Image of the .fits file
    hdr : dict()
        Header of the .fits file

    """
    if path:
        try:
            HDU = fits.open(image)[0]
        except:
            print(image)
            raise ValueError

        hdr = HDU.header
        img = HDU.data
    else:
        hdr = image.header
        img = image.data

    return img, hdr


def WCStoPIX(WCScoords, image):
    """
    
    Parameters
    ----------
    WCScoords : tuple or list
        WCS coordinates of an object with position x as the first element and y as the second element
    image : str or HDU
        Image to which we want to project the coordinates

    Returns
    -------
    PIXcoords : tuple
        Pixel coordinates projected on the image of an object with position x as the first element and y as the second element

    """

    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel

    x, y = WCScoords
    img, hdr = OpenImage(image)

    wcs = WCS(hdr)

    c = SkyCoord(x, y,
                 frame="icrs",
                 unit="deg")

    PIXcoords = skycoord_to_pixel(coords=c,
                                  wcs=wcs,
                                  origin=0,
                                  mode='all')
    return PIXcoords


def PIXtoWCS(PIXcoords, image):
    """
    
    Parameters
    ----------
    PIXcoords : tuple or list
        Pixel coordinates of an object with position x as the first element and y as the second element
    image : str or HDU
        Image to which we want to project the coordinates

    Returns
    -------
    WCScoords : tuple
        WCS coordinates projected on the image of an object with position x as the first element and y as the second element

    """

    from astropy.wcs import WCS
    from astropy.wcs.utils import pixel_to_skycoord

    x, y = PIXcoords
    img, hdr = OpenImage(image)

    wcs = WCS(hdr)

    coords = pixel_to_skycoord(xp=x, yp=y,
                               wcs=wcs,
                               origin=0,
                               mode='all')

    WCScoords = [coords.ra.degree, coords.dec.degree]

    return WCScoords


def Convolve(image, pixsize, fwhm):
    """
    

    Parameters
    ----------
    image : str or fits
        Image to convolve
    pixsize : float
        Original size of pixel in arcsec
    fwhm : float
        FWHM of the convolution beam

    Returns
    -------
    smoothed : fits
        Smoothed fits with image and new header

    """

    from astropy.convolution import convolve, Gaussian2DKernel

    img, hdr = OpenImage(image)

    coef = 2 * np.sqrt(2 * np.log(2))
    x_stddev = fwhm / pixsize / coef
    y_stddev = fwhm / pixsize / coef
    print("sigma in pixel : ", x_stddev)

    smoothed = convolve(img, Gaussian2DKernel(x_stddev=x_stddev,
                                              y_stddev=y_stddev))

    # hdr["Beam"] = x_stddev
    smoothed = fits.PrimaryHDU(data=smoothed, header=hdr)

    return smoothed


def orderingPoints(points, dist=1):
    xp, yp = points[0]
    remaining_coords = points.copy().tolist()
    order = []
    k = 0

    while k < len(points):
        for i, (x, y) in enumerate(remaining_coords):
            if (x - xp) ** 2 + (y - yp) ** 2 <= dist:
                order.append([x, y])
                remaining_coords.remove(remaining_coords[i])
                xp, yp = x, y
                k += 1
    return order


def Window(image, coord="wcs"):
    import scipy.ndimage.morphology as snm
    from shapely.geometry import Polygon
    from polygons import ReshapeCoordForPoly

    img, hdr = OpenImage(image)

    filled_mask = snm.binary_fill_holes(img != np.nan)
    interior_mask = snm.binary_erosion(filled_mask)
    edges = filled_mask ^ interior_mask
    mask_idx = np.flip(np.where(edges), axis=0)

    points = ReshapeCoordForPoly(mask_idx[0], mask_idx[1])

    ordered = orderingPoints(points, dist=1)

    if coord == "pix":
        return Polygon(ordered)

    if coord == "wcs":
        x, y = PIXtoWCS(np.reshape(np.ravel(ordered), (2, len(mask_idx[0])), order='F'), image)
        return Polygon(ReshapeCoordForPoly(x, y))

def LinearInterpolationPolygon(xo, yo, n=100):
    """

    Parameters
    ----------
    xo : list
        x coordinates of polygon
    yo : list
        y coordinates of polygon
    n : int, optional
        Number of points for linear interpolation between two points. The default is 100.

    Returns
    -------
    x : list
        x coordinates of interpolated polygon
    y : list
        y coordinates of interpolated polygon

    """

    x_new = []
    y_new = []
    xlen = len(xo)

    for i, xvalue in enumerate(xo):
        dx = xo[(i + 1) % xlen] - xvalue
        dy = yo[(i + 1) % xlen] - yo[i]

        xint = np.linspace(0, dx, n)
        yint = np.linspace(0, dy, n)

        [x_new.append(int(np.round(xvalue + xi))) for xi in xint]
        [y_new.append(int(np.round(yo[i] + yi))) for yi in yint]

    x = [xx for xx, yy in list(set(zip(x_new, y_new)))]
    y = [yy for xx, yy in list(set(zip(x_new, y_new)))]

    return x, y

def PolygonToMask(polygons, image, n=100, verbose=False):
    """

    Parameters
    ----------
    polygons : list of shapely polygon
        Polygons to mask
    image : str or fits
        Path to the image or fits object to put the mask on
    verbose : bool, optional
        The default is False.

    Returns
    -------
    mask : fits object
        Fits image with the masks of polygons.

    """

    # import shapely.geometry.polygon as shp
    from tqdm import tqdm
    import scipy.ndimage.morphology as snm
    import time

    img, hdr = OpenImage(image)

    xlen, ylen = np.shape(img)

    mask = np.zeros_like(img)

    if verbose:
        print("Creating mask for ellipses \n")
        time.sleep(1)
    for polygon in tqdm(polygons, disable=not verbose):

        WCScoords = polygon.exterior.xy
        x, y = WCStoPIX(WCScoords, image)
        x, y = LinearInterpolationPolygon(x, y, n=n)

        for xx, yy in zip(x, y):
            try:
                mask[yy, xx] = 1
            except:
                continue

    filled_mask = snm.binary_fill_holes(mask) * 1

    mask = fits.PrimaryHDU(data=filled_mask, header=hdr)
    # mask.writeto(name+".fits",overwrite=True)
    return mask

def MeanPixelInPolygon(polygons, image, verbose=False):
    """

    Parameters
    ----------
    polygons : list of shapely polygons
        Polygons in which to compute the average of the pixels
    image : str or fits object
        fits image that have the pixels values
    verbose : bool, optional
        The default is False.

    Returns
    -------
    meanpix : list
        i_th position gives the average value of pixel of the polygon_i

    """

    from tqdm import tqdm

    meanpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            meanpix.append(np.nan)
        else:
            log_valu = np.nanmean(np.log(values))
            meanpix.append(np.exp(log_valu))

    return meanpix

def MedPixelInPolygon(polygons, image, verbose=False):
    """

    Parameters
    ----------
    polygons : list of shapely polygons
        Polygons in which to compute the average of the pixels
    image : str or fits object
        fits image that have the pixels values
    verbose : bool, optional
        The default is False.

    Returns
    -------
    meanpix : list
        i_th position gives the average value of pixel of the polygon_i

    """

    from tqdm import tqdm

    meanpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            meanpix.append(np.nan)
        else:
            meanpix.append(np.nanmedian(values))

    return meanpix


def MaxPixelInPolygon(polygons, image, verbose=False):
    """
    
    Parameters
    ----------
    polygons : list of shapely polygons
        Polygons in which to compute the max of the pixels
    image : str or fits object
        fits image that have the pixels values
    verbose : bool, optional
        The default is False.

    Returns
    -------
    maxpix : list
        i_th position gives the maximum value of pixel of the polygon_i
    """

    from tqdm import tqdm

    maxpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            maxpix.append(np.nan)
        else:
            maxpix.append(np.nanmax(values))

    return maxpix


def pixelBins(catalog, fitsfile, Poly_key = "polygon", norm=True):
    import scipy.ndimage.morphology as snm
    from . import multiscale_structures as ms
    
    image, header = OpenImage(fitsfile)

    Hlst, Llst, Ilst, HLlst, HIlst, LIlst = [], [], [], [], [], []

    Nlst = []

    ###### Create mask for each type of structure
    maskH = np.zeros_like(image)
    maskL = np.zeros_like(image)
    maskI = np.zeros_like(image)

    for idx, poly in enumerate(catalog[Poly_key]):

        x, y = WCStoPIX(poly.exterior.xy, fitsfile)
        x, y = LinearInterpolationPolygon(x, y)

        if catalog["mode"][idx] == ms.StructureMode.HIERARCHICAL.name:
            for x, y in zip(x, y):
                try:
                    maskH[y, x] = 1
                except IndexError:
                    pass
        elif catalog["mode"][idx] == ms.StructureMode.LINEAR.name:
            for x, y in zip(x, y):
                try:
                    maskL[y, x] = 1
                except IndexError:
                    pass
        elif catalog["mode"][idx] == ms.StructureMode.ISOLATED.name:
            for x, y in zip(x, y):
                try:
                    maskI[y, x] = 1
                except IndexError:
                    pass

    filled_maskH = snm.binary_fill_holes(maskH)
    filled_maskL = snm.binary_fill_holes(maskL)
    filled_maskI = snm.binary_fill_holes(maskI)

    logmin = np.log10(np.amin(image))
    logmax = np.log10(np.amax(image))
    logbins = np.logspace(logmin, logmax)
    print(f"decomposing using {len(logbins)} bins evenly log-sampled between {logmin} and {logmax} [pixel unit]")

    for k, threshold in enumerate(logbins[1:]):
        im = np.logical_and(image <= threshold, image > logbins[k])
        N = np.nansum(im)

        common_HL = filled_maskH * filled_maskL * im
        common_LI = filled_maskI * filled_maskL * im
        common_HI = filled_maskH * filled_maskI * im

        HL = np.nansum(common_HL)
        LI = np.nansum(common_LI)
        HI = np.nansum(common_HI)

        H = np.nansum((filled_maskH * im) ^ common_HL ^ common_HI)
        L = np.nansum((filled_maskL * im) ^ common_HL ^ common_LI)
        I = np.nansum((filled_maskI * im) ^ common_HI ^ common_LI)

        if norm:
            Hlst.append(H / N)
            Llst.append(L / N)
            Ilst.append(I / N)

            HLlst.append(HL / N)
            LIlst.append(LI / N)
            HIlst.append(HI / N)
        else:
            Hlst.append(H)
            Llst.append(L)
            Ilst.append(I)

            HLlst.append(HL)
            LIlst.append(LI)
            HIlst.append(HI)

        Nlst.append(N)

    return Nlst, Hlst, Llst, Ilst, HLlst, LIlst, HIlst, logbins

def _pixelBins(graph, fitsfile):
    import scipy.ndimage.morphology as snm

    image, header = OpenImage(fitsfile)

    Hlst, Llst, Ilst, HLlst, HIlst, LIlst = [], [], [], [], [], []

    Nlst = []

    X, Y = np.shape(image)

    ###### Create mask for each type of structure
    maskH = np.zeros_like(image)
    maskL = np.zeros_like(image)
    maskI = np.zeros_like(image)

    for node, poly in graph.nodes("Polygon"):

        x, y = WCStoPIX(poly.exterior.xy, fitsfile)
        x, y = LinearInterpolationPolygon(x, y)

        if graph.nodes("Mode")[node] == "Hierarchical":
            for x, y in zip(x, y):
                try:
                    maskH[y, x] = 1
                except IndexError:
                    pass
        if graph.nodes("Mode")[node] == "Linear":
            for x, y in zip(x, y):
                try:
                    maskL[y, x] = 1
                except IndexError:
                    pass
        if graph.nodes("Mode")[node] == "Isolated":
            for x, y in zip(x, y):
                try:
                    maskI[y, x] = 1
                except IndexError:
                    pass

    filled_maskH = snm.binary_fill_holes(maskH)
    filled_maskL = snm.binary_fill_holes(maskL)
    filled_maskI = snm.binary_fill_holes(maskI)

    logmin = np.log10(np.amin(image))
    logmax = np.log10(np.amax(image))
    logbins = np.logspace(logmin, logmax)

    for k, threshold in enumerate(logbins[1:]):
        im = np.logical_and(image <= threshold, image > logbins[k])
        N = np.nansum(im)

        common_HL = filled_maskH * filled_maskL * im
        common_LI = filled_maskI * filled_maskL * im
        common_HI = filled_maskH * filled_maskI * im

        HL = np.nansum(common_HL)
        LI = np.nansum(common_LI)
        HI = np.nansum(common_HI)

        H = np.nansum((filled_maskH * im) ^ common_HL ^ common_HI)
        L = np.nansum((filled_maskL * im) ^ common_HL ^ common_LI)
        I = np.nansum((filled_maskI * im) ^ common_HI ^ common_LI)

        Hlst.append(H / N)
        Llst.append(L / N)
        Ilst.append(I / N)

        HLlst.append(HL / N)
        LIlst.append(LI / N)
        HIlst.append(HI / N)

        Nlst.append(N)

    return Nlst, Hlst, Llst, Ilst, HLlst, LIlst, HIlst, logbins
