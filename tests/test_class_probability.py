"""Regression tests for an explicit scalar classifier-probability target."""
from unittest.mock import patch
import numpy as np
import numpy.testing as npt
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              BaggingClassifier)
import forestci.forestci as fci


def model_data(n_classes=3, labels=None):
    x, y = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=n_classes, n_clusters_per_class=1, random_state=91)
    if labels is not None:
        y = np.asarray(labels)[y]
    model = RandomForestClassifier(
        n_estimators=90, max_depth=2, random_state=17).fit(x, y)
    return x, y, model


def reference(model, x, index):
    pred = np.array([t.predict_proba(x)[:, index]
                     for t in model.estimators_]).T
    centered = pred - pred.mean(axis=1, keepdims=True)
    inbag = fci.calc_inbag(len(x), model)
    raw = np.array([sum(np.mean((counts - 1) * row) ** 2
                        for counts in inbag) for row in centered])
    correction = (len(x) * np.var(inbag, axis=1).mean()
                  * np.mean(centered ** 2, axis=1) / model.n_estimators)
    return raw - correction, pred, centered


@pytest.mark.parametrize("memory_constrained", [False, True])
@pytest.mark.parametrize("class_index", [0, 1, 2])
def test_probability_matches_independent_reference(memory_constrained,
                                                  class_index):
    x, _, model = model_data(labels=["pear", "apple", "orange"])
    expected, pred, centered = reference(model, x, class_index)
    npt.assert_allclose(pred.mean(axis=1),
                        model.predict_proba(x)[:, class_index])
    npt.assert_allclose(fci._centered_prediction_forest(
        model, x, class_index=class_index), centered)
    actual = fci.random_forest_error(
        model, x.shape, x, calibrate=False, class_index=class_index,
        memory_constrained=memory_constrained,
        memory_limit=0.002 if memory_constrained else None)
    npt.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)


def test_soft_leaf_probability_differs_from_legacy_hard_votes():
    x, _, model = model_data(n_classes=2)
    _, pred, centered = reference(model, x, 1)
    hard = np.array([t.predict(x) for t in model.estimators_]).T
    assert np.any((pred > 0) & (pred < 1))
    assert not np.allclose(pred.mean(axis=1), hard.mean(axis=1))
    legacy = fci._centered_prediction_forest(model, x)
    npt.assert_allclose(legacy, hard - hard.mean(axis=1, keepdims=True))
    assert not np.allclose(legacy, centered)


def test_corresponding_class_is_invariant_under_label_permutation():
    x, y, model = model_data()
    labels = np.array([30, 10, 20])
    other = RandomForestClassifier(
        n_estimators=90, max_depth=2, random_state=17).fit(x, labels[y])
    for index, label in enumerate(labels):
        mapped = np.flatnonzero(other.classes_ == label)[0]
        npt.assert_allclose(model.predict_proba(x)[:, index],
                            other.predict_proba(x)[:, mapped])
        first = fci.random_forest_error(
            model, x.shape, x, calibrate=False, class_index=index)
        second = fci.random_forest_error(
            other, x.shape, x, calibrate=False, class_index=mapped)
        npt.assert_allclose(first, second, atol=1e-14, rtol=1e-12)


def test_binary_complement_has_equal_variance():
    x, _, model = model_data(n_classes=2)
    first = fci.random_forest_error(
        model, x.shape, x, calibrate=False, class_index=0)
    second = fci.random_forest_error(
        model, x.shape, x, calibrate=False, class_index=np.int64(1))
    npt.assert_allclose(first, second, atol=1e-14)


def test_calibration_forwards_selector_to_half_forest():
    x, _, model = model_data()
    original = fci._centered_prediction_forest
    calls = []

    def checked(forest, xt, y_output=None, class_index=None):
        calls.append((forest.n_estimators, class_index))
        return original(forest, xt, y_output, class_index)

    state = np.random.get_state()
    try:
        np.random.seed(31)
        with patch.object(fci, "_centered_prediction_forest",
                          side_effect=checked), patch.object(
                              fci, "calibrateEB",
                              side_effect=lambda variances, _: variances):
            result = fci.random_forest_error(
                model, x.shape, x, class_index=2,
                memory_constrained=True, memory_limit=0.002)
    finally:
        np.random.set_state(state)
    assert calls == [(90, 2), (45, 2)]
    assert result.shape == (len(x),)


@pytest.mark.parametrize("index", [-1, 3, 0.5, True, np.bool_(False), "1"])
def test_invalid_index(index):
    x, _, model = model_data()
    with pytest.raises(ValueError, match="class_index must index"):
        fci.random_forest_error(
            model, x.shape, x, calibrate=False, class_index=index)


def test_unsupported_estimators_and_selector_combinations():
    x, y, model = model_data()
    reg = RandomForestRegressor(n_estimators=5, random_state=8).fit(x, y)
    bag = BaggingClassifier(n_estimators=5, random_state=8).fit(x, y)
    for unsupported in [reg, bag]:
        with pytest.raises(ValueError, match="forest classifier"):
            fci.random_forest_error(
                unsupported, x.shape, x, class_index=0, calibrate=False)
    multi = RandomForestClassifier(n_estimators=5, random_state=8).fit(
        x, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="multioutput classifiers"):
        fci.random_forest_error(
            multi, x.shape, x, class_index=0, calibrate=False)
    with pytest.raises(ValueError, match="combined with y_output"):
        fci.random_forest_error(
            model, x.shape, x, class_index=0, y_output=0, calibrate=False)


def test_single_sample_centering():
    x, _, model = model_data()
    npt.assert_allclose(fci._centered_prediction_forest(
        model, x[0], class_index=1), fci._centered_prediction_forest(
            model, x[:1], class_index=1))
