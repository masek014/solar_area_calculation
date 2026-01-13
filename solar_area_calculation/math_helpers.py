from collections import namedtuple

import numpy as np


Point = namedtuple('Point', ['x', 'y', 'z'])


class Polygon():
    '''Represents a polygon as a collection of Point objects.'''

    def __init__(self, vertices: tuple[Point]):
        if len(vertices) < 3:
            raise ValueError(
                'Polygon object needs at least 3 points for construction.')
        self.vertices = vertices

    def compute_area(self) -> float:
        '''
        Compute the area of the polygon.

        WARNING: the current implementation of this method assumes that the
        polygon's vertices are ordered clockwise. If they are not ordered
        clockwise, then the area will be WRONG. In the future, the class
        __init__ should include a sorting algorithm that orders the corners.
        '''
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
        result = dot_product(
            Point(*total), unit_normal(self.vertices[0], self.vertices[1], self.vertices[2]))

        return abs(result / 2)


def determinant(A: list | np.ndarray) -> float:
    '''Compute the determinant of 3x3 matrix A.
    Taken from https://stackoverflow.com/a/12643315
    '''

    return A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + \
        A[0][2]*A[1][0]*A[2][1] - A[0][2]*A[1][1]*A[2][0] - \
        A[0][1]*A[1][0]*A[2][2] - A[0][0]*A[1][2]*A[2][1]


def unit_normal(a: Point, b: Point, c: Point) -> Point:
    '''Compute the unit normal vector of a plane defined by
    points a, b, and c.
    '''

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


def dot_product(a: Point, b: Point) -> float:
    '''Compute the dot product of 3-dimensional vectors.'''
    return a.x*b.x + a.y*b.y + a.z*b.z


def cross_product(a: Point, b: Point) -> Point:
    '''Compute the cross product of 3-dimensional vectors.'''
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x

    return Point(x, y, z)
