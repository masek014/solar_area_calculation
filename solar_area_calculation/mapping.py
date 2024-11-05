import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import os
import sunpy.data.sample
import sunpy.map

from matplotlib.animation import FuncAnimation


MAP_STYLE = 'map.mplstyle'
MODULE_DIR = os.path.dirname(__file__)
MAP_STYLE = os.path.join(MODULE_DIR, 'map.mplstyle')

plt.style.use(MAP_STYLE)


def example_aia_map() -> sunpy.map.GenericMap:
    """
    Generates an example AIA map for plotting purposes.
    """

    aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    aia_map.plot_settings['cmap'] = 'inferno_r'
    
    return aia_map


def gen_blank_map() -> sunpy.map.GenericMap:
    """
    Creates a blank map for plotting a region on.

    It uses the map from example_aia_map() but then set all pixel data to zero.
    This is done so that the coordinate frame of the AIA map can be used.
    """

    map_ = example_aia_map()
    map_.data[:] = np.zeros(map_.data.shape)

    return map_


def plot_sphere(
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D,
    resolution: int = 20,
    radius: float = 1,
    **kwargs: dict
) -> mpl_toolkits.mplot3d.art3d.Line3DCollection:
    """
    Plot a sphere, representative of the Sun.
    """

    default_kwargs = dict(color='red', lw=0.5, alpha=0.4)
    kwargs = {**default_kwargs, **kwargs}

    uu, vv = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = radius * np.cos(uu)*np.sin(vv)
    y = radius * np.sin(uu)*np.sin(vv)
    z = radius * np.cos(vv)
    s = ax.plot_wireframe(x, y, z, **kwargs)

    return s


def sphere_mosaic(
    fig: plt.Figure,
    ax: plt.Axes,
    out_file: str,
    fps: int = 10,
    elevations: np.ndarray = np.linspace(45, -45, 5),
    azimuths: np.ndarray = np.linspace(-45, 45, 20)
):
    """
    Make a movie iterating through various perspectives
    of the 3D scene in fig, ax. 
    """

    elevs, azims = np.meshgrid(elevations, azimuths)
    elevs = elevs.T
    azims = azims.T
    azims[1::2, :] = azims[1::2, ::-1] # Reverse every-other row
    angles = np.vstack([elevs.ravel(), azims.ravel()]).T
    animate = lambda i: ax.view_init(*angles[i])

    anim = FuncAnimation(fig, animate,
        frames=len(angles), interval=20)
    anim.save(out_file, fps=fps)

    print(f'Sphere mosaic saved to {out_file}')
