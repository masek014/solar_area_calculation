# Solar Areas

## Introduction

This little repository houses a relatively simple solution I came up with in order to estimate the solar surface area contained within a 2D shape.
The motivation for developing this was to estimate the solar surface area visible within an instrument's field of view.

The implemented solution is capable of computing areas for *any* `SkyRegion` class, as defined by the [Astropy `regions`](https://astropy-regions.readthedocs.io/en/stable/#) package.
See a list of defined `SkyRegion` classes [here](https://astropy-regions.readthedocs.io/en/stable/shapes.html).
It also requires the use of a Sunpy `Map` object for observer properties to determine scaling.
Perhaps this can be changed in the future to remove this dependence by using `Observer` objects directly?

There are plans to further improve the code structure.


## Methodology

The method is as follows:
1. Determine the bounding box of the region of interest.
2. Generate a grid of points within the bounding box with a specified resolution.
3. Filter out all points within the bounding box that lie outside of the region of interest.
4. Project the points onto the solar sphere using Astropy `SkyCoord` methods. The coordinates are projected into the Heliographic Stonyhurst frame as defined by `sunpy.coordinates.frames.HeliographicStonyhurst`, which provides the coordinates as longitude-latitude pairs. This step also removes all in-region points that do not lie on the solar sphere.
5. Construct a polygon using neighboring points. A polygon will be given four points if four points are available, but it will use three points if only three are available.
6. Compute the area of the polygon.
7. Repeat for all 3- or 4-point polygons possible.
8. Sum all areas to get an estimate of the total area.

Note that since this method uses polygons to estimate the area, it will generally *underestimate* the actual area.
Additionally, it is governed by the resolution and scale of the provided Sunpy `Map` since the coordinates are converted between `SkyCoord` objects and `PixCoord` objects when filtering out the points that are external to the region of interest.


## Examples
See the `examples.ipynb` notebook for some examples of usage with several `SkyRegion` objects.


## Diagnostics

Some simple tests were performed to gauge performance of the method.
