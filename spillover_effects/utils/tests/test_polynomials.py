import numpy as np
import pytest
from spillover_effects.utils.polynomials import (
    poly_identity,
    poly_interact,
    poly_cubic,
    poly_square,
)


@pytest.fixture
def sample_data():
    t = np.array([1, 2, 3])
    r = np.array([4, 5, 6])
    return t, r


def test_poly_identity(sample_data):
    t, r = sample_data
    result = poly_identity(t, r)
    expected = np.column_stack((t, r))
    assert np.array_equal(result, expected)


def test_poly_interact(sample_data):
    t, r = sample_data
    result = poly_interact(t, r)
    expected = np.column_stack((t, r, t * r))
    assert np.array_equal(result, expected)


def test_poly_cubic(sample_data):
    t, r = sample_data
    result = poly_cubic(t, r)
    expected = np.column_stack((t, t**2, t**3, r, r**2, r**3, t * r))
    assert np.array_equal(result, expected)


def test_poly_square(sample_data):
    t, r = sample_data
    result = poly_square(t, r)
    expected = np.column_stack((t, t**2, r, r**2, t * r))
    assert np.array_equal(result, expected)
