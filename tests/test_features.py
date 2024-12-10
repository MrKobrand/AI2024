import pandas as pd
from src.features import compute_minkowski_distances, compute_euclidean_distances, compute_manhattan_distances, compute_cosine_distances, compute_jensenshannon_distances


def test_compute_minkowski_distances():
    X = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })

    expected_result = pd.DataFrame([[0, 1.41421356], [1.41421356, 0]])
    result = compute_minkowski_distances(X, p=2)
    pd.testing.assert_frame_equal(result, expected_result)


def test_compute_euclidean_distances():
    X = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })

    expected_result = pd.DataFrame([[0, 1.41421356], [1.41421356, 0]])
    result = compute_euclidean_distances(X)
    pd.testing.assert_frame_equal(result, expected_result)


def test_compute_manhattan_distances():
    X = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })

    expected_result = pd.DataFrame([[0, 2.0], [2.0, 0]])
    result = compute_manhattan_distances(X)
    pd.testing.assert_frame_equal(result, expected_result)


def test_compute_cosine_distances():
    X = pd.DataFrame({
        'A': [1, 0],
        'B': [0, 1]
    })

    expected_result = pd.DataFrame([[0, 1.0], [1.0, 0]])
    result = compute_cosine_distances(X)
    pd.testing.assert_frame_equal(result, expected_result)


def test_compute_jensenshannon_distances():
    X = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })

    expected_result = pd.DataFrame([[0, 0.06490450], [0.06490450, 0]])
    result = compute_jensenshannon_distances(X)
    pd.testing.assert_frame_equal(result, expected_result)
