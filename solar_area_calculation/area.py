import itertools
import multiprocessing as mp
import typing

import astropy.units as u
import numpy as np
import sunpy.map

from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion, SkyRegion, PixCoord
from sunpy.coordinates import frames

from . import math_helpers

SOLAR_RADIUS_METERS = 6.957e8 * u.meter
SOLAR_RADIUS_ARCSECONDS = 959.63 * u.arcsecond
SOLAR_AREA_METERS = 6.082104402130212E18 * (u.meter)**2


'''
NOTE: I avoided using Quantity objects when handling coordinates and polygons.
Adding them in drastically increases computation time, so it's best to add
after the fact.
'''


def get_region_points(
    map_: sunpy.map.GenericMap,
    region: SkyRegion,
    resolution: int
) -> np.ndarray:
    '''Generates a grid of points within the specified region.
    resolution specifies the number of points in a SINGLE dimension,
    i.e., you will end up with a total of resolution**2 number of
    coordinate pairs.
    '''
    bbox = region.to_pixel(map_.wcs).bounding_box
    corners = [
        (bbox.ixmin, bbox.iymin),  # bottom left
        (bbox.ixmin, bbox.iymax),  # top left
        (bbox.ixmax, bbox.iymin),  # bottom right
        (bbox.ixmax, bbox.iymax)  # top right
    ]
    bl = PixCoord(*corners[0]).to_sky(map_.wcs)
    tr = PixCoord(*corners[-1]).to_sky(map_.wcs)

    x_range = (bl.Tx.value, tr.Tx.value)
    y_range = (bl.Ty.value, tr.Ty.value)

    x = np.linspace(*x_range, resolution) * bl.Tx.unit
    y = np.linspace(*y_range, resolution) * bl.Ty.unit

    X, Y = np.meshgrid(x, y)
    coord_pairs = np.vstack([X.ravel(), Y.ravel()]).T

    # Now screen out the points external to the region.
    region_pix = region.to_pixel(map_.wcs)
    x, y = SkyCoord(
        coord_pairs, frame=map_.coordinate_frame).to_pixel(map_.wcs)
    pixcoords = PixCoord(x, y)

    inds = region_pix.contains(pixcoords)
    coord_pairs[~inds] = np.array([np.nan, np.nan])

    return coord_pairs


def convert_xy_to_lonlat(
    map_: sunpy.map.GenericMap,
    coord_pairs: typing.Iterable[tuple[u.Quantity, u.Quantity]]
) -> np.ndarray:
    '''Uses the map coordinate frame to transform the (x,y) pairs of coord_pairs
    into (longitude,latitude) pairs on the solar disk.
    This step also filters out all off-disk points.
    '''
    coords = SkyCoord(coord_pairs, frame=map_.coordinate_frame)
    coords_hgs = coords.transform_to(frames.HeliographicStonyhurst)
    lonlat_pairs = np.vstack((coords_hgs.lon.radian, coords_hgs.lat.radian)).T

    return lonlat_pairs


def get_lonlat_pairs(
    map_: sunpy.map.GenericMap,
    region: RectangleSkyRegion,
    resolution: int
) -> np.ndarray:
    '''Computes (longitude,latitude) coordinate pairs in radians
    within the region. These are used to represent the region
    as polygons, from which the area is computed.
    resolution specifies the number of points in a SINGLE dimension,
    i.e., you will end up with a total of resolution**2 number of
    coordinate pairs.
    '''
    coord_pairs = get_region_points(map_, region, resolution)
    num_jobs = mp.cpu_count()
    pair_chunks = np.array_split(coord_pairs, num_jobs)

    # Parallelizing can significantly increase speed for high resolution.
    # For low resolution (~100) it is comparable or even worse than serial,
    # but it's so fast for low resolution that it doesn't even matter.
    with mp.Pool(processes=num_jobs) as p:
        out = p.starmap(convert_xy_to_lonlat, zip(
            itertools.repeat(map_), pair_chunks))

    lonlat_pairs = np.concatenate(out)

    return lonlat_pairs


def compute_chunk_area(chunk: np.ndarray) -> float:
    '''Computes the polygon area of the provided chunk.'''
    area = 0
    poly_coords = np.lib.stride_tricks.sliding_window_view(
        chunk, (2, 2, 3)
    )[:, :, 0]

    # Reshape so each element is an array of polygon corners.
    # Each element has contains the four corners of the polygon.
    rows, cols = poly_coords.shape[0:2]
    poly_coords = poly_coords.reshape(rows-1*cols-1, 4, 3)
    for coords in poly_coords:
        coords[[2, 3]] = coords[[3, 2]]  # Make corners clockwise
        bad = np.any(np.isnan(coords), axis=1)
        coords = coords[~bad, :]  # Drop any coords with nan
        if len(coords) > 2:
            points = [math_helpers.Point(*p) for p in coords]
            poly = math_helpers.Polygon(points)
            area += poly.compute_area()

    return area


def compute_solar_area(
    map_: sunpy.map.GenericMap,
    region: RectangleSkyRegion,
    resolution: int,
    return_coordinates: bool = False
) -> u.Quantity | tuple[u.Quantity, np.ndarray]:
    '''Resolution determines the number of grid points in each direction,
    resulting in a total of resolution**2 number of points.

    If return_coordinates is True, then a tuple is returned containing the area
    and the coordinates of the grid.

    The area scaling is determined by the radius of the Sun as stored
    in the coordinate frame of the map_ object.

    Also note that the accuracy of the area is dependent on the grid resolution
    of the map itself.
    '''

    lonlat_pairs = get_lonlat_pairs(map_, region, resolution)
    lon = lonlat_pairs[:, 0]
    lat = np.pi/2 - lonlat_pairs[:, 1]  # Move zero to the observer-Sun line

    # Reshape into a meshgrid-like structure.
    lon = np.reshape(lon, (resolution, resolution))
    lat = np.reshape(lat, (resolution, resolution))

    # Compute the cartesian coordinates of the points on the sphere.
    # NOTE: the sphere is given a radius of 1 here so we can apply
    # the scale after the computation (quicker).
    X = np.sin(lat) * np.cos(lon)
    Y = np.sin(lat) * np.sin(lon)
    Z = np.cos(lat)
    coords = np.dstack([X, Y, Z])

    # Define the chunks for which the areas will be computed.
    # They have shapes (2, resolution, 3), so the two rows represent
    # two neighboring rows of grid points that are used for computing
    # the polygons.
    slices = np.lib.stride_tricks.sliding_window_view(
        coords, (1, *coords.shape[1:3])
    )[:, 0, 0]
    chunks = []
    for i in range(resolution-1):
        chunks.append(np.concatenate(slices[i:i+2]))

    with mp.Pool(processes=mp.cpu_count()) as p:
        out = p.map(compute_chunk_area, chunks)
    area = np.sum(out) * (map_.coordinate_frame.rsun**2)

    if return_coordinates:
        ret = area, coords
    else:
        ret = area

    return ret
