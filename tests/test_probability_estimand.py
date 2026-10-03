"""A trained model with a reproducible hard-vote / probability mismatch."""
import numpy as np
import numpy.testing as npt
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.exceptions import NotFittedError
import forestci.forestci as fci


def test_root_only_forest_has_probability_uncertainty_despite_constant_votes():
    x = np.arange(80, dtype=float).reshape(-1, 1)
    y = np.repeat([0, 1], [8, 72])
    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=80, random_state=42).fit(x, y)
    inbag = fci.calc_inbag(len(x), forest)
    # Every tree is a single leaf. Its probability is exactly the bootstrap
    # proportion of class 1, independently recoverable from inbag counts.
    proportions = y @ inbag / len(x)
    probabilities = np.array([
        tree.predict_proba(x[:1])[0, 1] for tree in forest.estimators_])
    npt.assert_allclose(probabilities, proportions)
    assert np.all(np.array([tree.predict(x[:1])[0]
                            for tree in forest.estimators_]) == 1)
    assert np.ptp(probabilities) > 0
    centered = proportions - proportions.mean()
    raw = sum(np.mean((counts - 1) * centered) ** 2 for counts in inbag)
    correction = (len(x) * np.var(inbag, axis=1).mean()
                  * np.mean(centered ** 2) / forest.n_estimators)
    expected = raw - correction
    assert expected > 0
    legacy = fci.random_forest_error(forest, x.shape, x[:1], calibrate=False)
    selected = fci.random_forest_error(
        forest, x.shape, x[:1], class_index=1, calibrate=False)
    npt.assert_array_equal(legacy, [0])
    npt.assert_allclose(selected, [expected], atol=1e-14)
    npt.assert_allclose(forest.predict_proba(x[:1])[0, 1], proportions.mean())


def test_unfitted_selector_raises_standard_not_fitted_error():
    with pytest.raises(NotFittedError):
        fci.random_forest_error(
            RandomForestClassifier(), (4, 1), np.zeros((1, 1)), class_index=0)


def test_bootstrapped_extra_trees_probability_and_single_class():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.repeat(["only"], len(x))
    forest = ExtraTreesClassifier(
        n_estimators=30, bootstrap=True, random_state=6).fit(x, y)
    centered = fci._centered_prediction_forest(forest, x, class_index=0)
    npt.assert_array_equal(centered, np.zeros((20, 30)))
    npt.assert_array_equal(fci.random_forest_error(
        forest, x.shape, x, class_index=0, calibrate=False), np.zeros(20))
