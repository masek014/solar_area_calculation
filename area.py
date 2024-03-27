import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import multiprocessing as mp
import numpy as np
import sunpy.data.sample
import sunpy.map
import typing

from astropy.coordinates import SkyCoord
from collections import namedtuple
from itertools import repeat
from matplotlib.animation import FuncAnimation
from regions import RectangleSkyRegion, SkyRegion, PixCoord
from sunpy.coordinates import frames


SOLAR_RADIUS_METERS = 6.957e8 * u.meter
SOLAR_RADIUS_ARCSECONDS = 959.63 * u.arcsecond
SOLAR_AREA_METERS = 6.082104402130212E18 * (u.meter)**2


"""
NOTE: I avoided using Quantity objects when handling coordinates and polygons.
Adding them in drastically increases computation speed, so it's best to add
after the fact.
"""

Point = namedtuple('Point', ['x', 'y', 'z'])


class Polygon():
    """
    Represents a polygon as a collection of Point objects.
    """
    
    def __init__(self, vertices: tuple[Point]):
        
        if len(vertices) < 3:
            raise ValueError('Polygon object needs at least 3 points for construction.')
        self.vertices = vertices


    def compute_area(self) -> float:
        """
        Compute the area of the polygon.

        WARNING: the current implementation of this method assumes that the
        polygon's vertices are ordered clockwise. If they are not ordered
        clockwise, then the area will be WRONG. In the future, the class
        __init__ should include a sorting algorithm that orders the corners.
        """
    
        total = [0, 0, 0]
        for i in range(len(self.vertices)):
            vi1 = self.vertices[i]
            if i is len(self.vertices)-1:
                vi2 = self.vertices[0]
            else:
                vi2 = self.vertices[i+1]
            prod = cross_product(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = dot_product(Point(*total), unit_normal(self.vertices[0], self.vertices[1], self.vertices[2]))
        
        return abs(result / 2)


def determinant(A: list | np.ndarray) -> float:
    """
    Compute the determinant of 3x3 matrix A.
    Taken from https://stackoverflow.com/a/12643315
    """
    
    return A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1] - A[0][2]*A[1][1]*A[2][0] - A[0][1]*A[1][0]*A[2][2] - A[0][0]*A[1][2]*A[2][1]


def unit_normal(a: Point, b: Point, c: Point) -> Point:
    """
    Compute the unit normal vector of a plane defined by points a, b, and c.
    """
    
    x = determinant([[1, a.y, a.z],
                     [1, b.y, b.z],
                     [1, c.y, c.z]])
    y = determinant([[a.x, 1, a.z],
                     [b.x, 1, b.z],
                     [c.x, 1, c.z]])
    z = determinant([[a.x, a.y, 1],
                     [b.x, b.y, 1],
                     [c.x, c.y, 1]])
    magnitude = (x**2 + y**2 + z**2)**0.5
    
    return Point(x / magnitude, y / magnitude, z / magnitude)


def dot_product(a: Point, b: Point) -> Point:
    """
    Compute the dot product of 3-dimensional vectors.
    """

    return a.x*b.x + a.y*b.y + a.z*b.z


def cross_product(a: Point, b: Point) -> Point:
    """
    Compute the cross product of 3-dimensional vectors.
    """
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x
    
    return Point(x, y, z)


def example_aia_map() -> sunpy.map.Map:
    """
    Generates an example AIA map for plotting purposes.
    """

    aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    aia_map.plot_settings['cmap'] = 'inferno_r'
    
    return aia_map


def gen_blank_map() -> sunpy.map.Map:
    """
    Creates a blank map for plotting a region on.

    It uses the map from example_aia_map() but then set all pixel data to zero.
    This is done so that the coordinate frame of the AIA map can be used.
    """

    map_ = example_aia_map()
    map_.data[:] = np.zeros(map_.data.shape)

    return map_


def plot_region(
    map_: sunpy.map.Map,
    region: SkyRegion,
    ax: matplotlib.axes.Axes,
    **kwargs: dict
):
    """
    Plots the provided region to the axis.
    """
    
    region.to_pixel(map_.wcs).plot(ax=ax, **kwargs)


def get_region_points(
    map_: sunpy.map.Map,
    region: SkyRegion,
    resolution: int
) -> np.ndarray:
    """
    Generates a grid of points within the specified region.

    resolution specifies the number of points in a SINGLE dimension,
    i.e., you will end up with a total of resolution**2 number of
    coordinate pairs.
    """

    bbox = region.to_pixel(map_.wcs).bounding_box
    corners = [
        (bbox.ixmin, bbox.iymin), # bottom left
        (bbox.ixmin, bbox.iymax), # top left
        (bbox.ixmax, bbox.iymin), # bottom right
        (bbox.ixmax, bbox.iymax)  # top right
    ]
    bl = PixCoord(*corners[0]).to_sky(map_.wcs)
    tr = PixCoord(*corners[-1]).to_sky(map_.wcs)

    x_range = (bl.Tx, tr.Tx)
    y_range = (bl.Ty, tr.Ty)
    
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    
    X, Y = np.meshgrid(x, y)
    coord_pairs = np.vstack([X.ravel(), Y.ravel()]).T

    # Now screen out the points external to the region.
    region_pix = region.to_pixel(map_.wcs)
    x, y = SkyCoord(coord_pairs, frame=map_.coordinate_frame).to_pixel(map_.wcs)
    pixcoords = PixCoord(x, y)

    inds = region_pix.contains(pixcoords)
    coord_pairs[~inds] = np.array([np.nan, np.nan])

    return coord_pairs


def convert_xy_to_lonlat(
    map_: sunpy.map.Map,
    coord_pairs: typing.Iterable[tuple[u.Quantity, u.Quantity]]
) -> np.ndarray:
    """
    Uses the map coordinate frame to transform the (x,y) pairs of coord_pairs
    into (longitude,latitude) pairs on the solar disk.

    This step also filters out all off-disk points.
    """

    coords = SkyCoord(coord_pairs, frame=map_.coordinate_frame)
    coords_hgs = coords.transform_to(frames.HeliographicStonyhurst)
    lonlat_pairs = np.vstack((coords_hgs.lon.radian, coords_hgs.lat.radian)).T

    return lonlat_pairs


def get_lonlat_pairs(
    map_: sunpy.map.Map,
    region: RectangleSkyRegion,
    resolution: int
) -> np.ndarray:
    """
    Computes (longitude,latitude) coordinate pairs in radians within the region.
    These are used to represent the region as polygons, from which
    the area is computed.

    resolution specifies the number of points in a SINGLE dimension,
    i.e., you will end up with a total of resolution**2 number of
    coordinate pairs.
    """

    coord_pairs = get_region_points(map_, region, resolution)

    num_jobs = mp.cpu_count()
    pair_chunks = np.array_split(coord_pairs, num_jobs)

    # Parallelizing can significantly increase speed for high resolution.
    # For low resolution (~100) it is comparable or even worse than serial,
    # but it's so fast for low resolution that it doesn't even matter.
    with mp.Pool(processes=num_jobs) as p:
        out = p.starmap(convert_xy_to_lonlat, zip(repeat(map_), pair_chunks))

    lonlat_pairs = np.concatenate(out)

    return lonlat_pairs


def compute_chunk_area(chunk: np.ndarray) -> float:
    """
    Computes the polygon area of the provided chunk.
    """

    area = 0
    poly_coords = np.lib.stride_tricks.sliding_window_view(
        chunk, (2, 2, 3)
    )[:, :, 0]

    # Reshape so each element is an array of polygon corners.
    # Each element has contains the four corners of the polygon.
    rows, cols = poly_coords.shape[0:2]
    poly_coords = poly_coords.reshape(rows-1*cols-1, 4, 3)

    for coords in poly_coords:

        coords[[2, 3]] = coords[[3, 2]] # Make corners clockwise
        bad = np.any(np.isnan(coords), axis=1)
        coords = coords[~bad,:] # Drop any coords with nan

        if len(coords) >= 3:
            points = [Point(*p) for p in coords]
            poly = Polygon(points)
            area += poly.compute_area()

    return area


def plot_sphere(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    resolution: int = 20,
    radius: float = 1,
    **kwargs: dict
) -> mpl_toolkits.mplot3d.art3d.Line3DCollection:

    default_kwargs = dict(
        color='red',
        lw=0.5,
        alpha=0.4
    )
    kwargs = {**default_kwargs, **kwargs}

    uu, vv = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = radius * np.cos(uu)*np.sin(vv)
    y = radius * np.sin(uu)*np.sin(vv)
    z = radius * np.cos(vv)
    s = ax.plot_wireframe(x, y, z, **kwargs)

    return s


def sphere_mosaic(fig, ax, out_file: str, fps: int = 10):

    elevations = np.linspace(45, -45, 5)
    azimuths = np.linspace(-45, 45, 20)
    elevs, azims = np.meshgrid(elevations, azimuths)
    elevs = elevs.T
    azims = azims.T
    azims[1::2, :] = azims[1::2, ::-1] # Reverse every-other row
    angles = np.vstack([elevs.ravel(), azims.ravel()]).T
    animate = lambda i: ax.view_init(*angles[i])

    anim = FuncAnimation(fig, animate,
        frames=len(angles), interval=20)
    anim.save(out_file, fps=fps)


def compute_solar_area(
    map_: sunpy.map.Map,
    region: RectangleSkyRegion,
    resolution: int,
    visualize: bool = False
) -> u.Quantity:
    """
    Resolution determines the number of grid points in each direction,
    resulting in a total of resolution**2 number of points.

    The area scaling is determined by the radius of the Sun as stored
    in the coordinate frame of the map_ object.

    Also note that the accuracy of the area is dependent on the grid resolution
    of the map itself.
    """

    lonlat_pairs = get_lonlat_pairs(map_, region, resolution)
    lon = lonlat_pairs[:,0]
    lat = np.pi/2 - lonlat_pairs[:,1] # Move zero to the observer-Sun line

    # Reshape into a meshgrid-like structure.
    lon = np.reshape(lon, (resolution, resolution))
    lat = np.reshape(lat, (resolution, resolution))

    # Compute the cartesian coordinates of the points on the sphere.
    # NOTE: the sphere is given a radius of 1 here so we can apply
    # the scale after the computation (quicker).
    r = np.ones(lon.shape)
    X = r * np.sin(lat) * np.cos(lon)
    Y = r * np.sin(lat) * np.sin(lon)
    Z = (r * np.cos(lat))
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

    # Just some visualization stuff.
    if visualize:

        fig = plt.figure(figsize=(5,5), constrained_layout=True)
        ax_map = plt.subplot(projection=map_)
        map_.plot(ax_map)
        map_.draw_limb(
            color='black',
            linewidth=1.25, linestyle='dotted'
        )
        ax_map.set(
            xlabel='X (arcsec)',
            ylabel='Y (arcsec)'
        )
        plot_region(map_, region, ax_map)
        ax_map.set_title('FOV on solar disk')

        # Plot the 3D surface and FOV.
        fig = plt.figure(figsize=(5,5), constrained_layout=True)
        ax_3d = plt.subplot(projection='3d')
        ax_3d.view_init(elev=0, azim=0)
        plot_sphere(ax_3d)
        ax_3d.set(
            xticks=[],
            ylabel='Solar radii',
            zlabel='Solar radii',
            xlim=(-1,1),
            ylim=(-1,1),
            zlim=(-1,1),
            box_aspect=[1,1,1]
        )
        # NOTE: The polygons represented by plot_surface
        # ARE NOT(!!!) the same as the polygons used to compute the
        # area. It's very expensive to store and plot those polygons
        # when the grid resolution is high.
        ax_3d.plot_surface(X, Y, Z, color='blue', alpha=1)
        ax_3d.grid(False)

        ax_3d.set(
            xticks=ax_3d.get_yticks(),
            xlabel='Solar radii',
            title='Rotate me!'
        )
        ax_3d.view_init(elev=15, azim=45)

        # Make a gif panning around the FOV.
        # This takes some time, hence why it's commented out.
        """fig, ax = plt.subplots(
            figsize=(8,8),
            constrained_layout=True,
            subplot_kw=dict(projection='3d')
        )
        ax.set(
            xlim=(-1,1),
            ylim=(-1,1),
            zlim=(-1,1),
            box_aspect=[1,1,1]
        )

        ax.plot_surface(X, Y, Z, color='blue', alpha=1)
        plot_sphere(ax, 100, alpha=0.2)
        ax.set_axis_off()
        sphere_mosaic(fig, ax, 'mosaic.gif')
        plt.close()"""

    return area