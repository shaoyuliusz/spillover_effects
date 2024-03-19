import numpy as np


def poly_identity(*args):
    return np.column_stack([*args])


def poly_interact(t, r):
    return np.column_stack((t, r, t * r))


def poly_cubic(t, r):
    return np.column_stack((t, t**2, t**3, r, r**2, r**3, t * r))


def poly_square(t, r):
    return np.column_stack((t, t**2, r, r**2, t * r))
