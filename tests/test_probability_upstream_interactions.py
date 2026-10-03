import copy
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.ensemble import RandomForestClassifier

import forestci as fci


def fitted_classifier():
    rng = np.random.RandomState(47)
    X = rng.normal(size=(60, 3))
    y = np.where(X[:, 0] + 0.4 * rng.normal(size=60) > 0, "yes", "no")
    model = RandomForestClassifier(
        n_estimators=8, max_depth=2, min_samples_leaf=5,
        bootstrap=True, random_state=42,
    ).fit(X, y)
    return X, model, fci.calc_inbag(len(X), model)


def test_probability_vector_matches_matrix_in_memory_mode():
    X, model, inbag = fitted_classifier()
    expected = fci.random_forest_error(
        model, X.shape, X[[1]], inbag=inbag,
        class_index=1, calibrate=False,
    )
    actual = fci.random_forest_error(
        model, X.shape, X[1], inbag=inbag,
        class_index=1, calibrate=False,
        memory_constrained=True, memory_limit=0.001,
    )
    assert actual.shape == (1,)
    assert_allclose(actual, expected)


@pytest.mark.parametrize("memory_constrained", [False, True])
def test_probability_calibration_reuses_selected_sampling_counts(memory_constrained):
    X, model, inbag = fitted_classifier()
    options = dict(
        class_index=1, memory_constrained=memory_constrained,
        memory_limit=0.001,
    )
    full = fci.random_forest_error(
        model, X.shape, X[:25], inbag=inbag, calibrate=False, **options
    )
    selected = np.array([6, 1, 4, 0])
    reduced = copy.deepcopy(model)
    reduced.estimators_ = [model.estimators_[i] for i in selected]
    reduced.n_estimators = len(selected)
    reduced_variance = fci.random_forest_error(
        reduced, X.shape, X[:25], inbag=inbag[:, selected],
        calibrate=False, **options
    )
    expected_noise = np.mean((reduced_variance - full) ** 2)
    with (
        patch("forestci.forestci.np.random.permutation", return_value=selected),
        patch("forestci.forestci.calc_inbag", side_effect=AssertionError("must reuse supplied counts")),
        patch("forestci.forestci.calibrateEB", side_effect=lambda values, noise: values) as calibrate,
    ):
        result = fci.random_forest_error(
            model, X.shape, X[:25], inbag=inbag, calibrate=True, **options
        )
    assert_allclose(result, full)
    assert calibrate.call_count == 1
    assert_allclose(calibrate.call_args.args[0], full)
    assert_allclose(calibrate.call_args.args[1], expected_noise)
