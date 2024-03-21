import numpy as np


def poly_identity(*args):
    return np.column_stack([*args])


def poly_interact(t, r):
    return np.column_stack((t, r, t * r))


def poly_cubic(t, r):
    if r is None:
        return np.column_stack((t, t**2, t**3))
    else:
        return np.column_stack((t, t**2, t**3, r, r**2, r**3, t * r))


def poly_square(t, r):
    if r is None:
        return np.column_stack((t, t**2))
    else:
        return np.column_stack((t, t**2, r, r**2, t * r))


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def sigmoid(x, a, k):
    """k controls kink, larger k means larger kink"""
    return 1 / (1 + np.exp(-k * (x - a)))


def middle_points(arr: np.ndarray) -> np.ndarray:
    """compute middle points of a given array
    Example usage:
    arr = np.linspace(0, 1, 6)
    result = middle_points(arr)
    print("Middle points:", result)
    """
    return (arr[:-1] + arr[1:]) / 2
