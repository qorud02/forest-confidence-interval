"""Integration checks for selected-column allocation and calibration."""
import gc
import weakref
import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import forestci.forestci as fci


def test_centering_releases_unselected_tree_probability_columns():
    references = []

    class Tree:
        def __init__(self, index):
            self.index = index

        def predict_proba(self, x):
            gc.collect()
            # Prior full probability matrices should no longer be retained
            # by the selected-column prediction list.
            assert all(reference() is None for reference in references)
            probabilities = np.full((len(x), 100), 0.01)
            probabilities[:, 7] += self.index * 0.001
            probabilities[:, 0] -= self.index * 0.001
            references.append(weakref.ref(probabilities))
            return probabilities

    class Forest(list):
        n_outputs_ = 1

    centered = fci._centered_prediction_forest(
        Forest(Tree(i) for i in range(5)), np.zeros((20, 1)), class_index=7)
    npt.assert_allclose(centered, np.tile(
        np.arange(-2, 3) * 0.001, (20, 1)), atol=1e-16)
    gc.collect()
    assert all(reference() is None for reference in references)


def test_actual_calibration_is_finite_for_probability_target():
    x, y = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=91)
    forest = RandomForestClassifier(
        n_estimators=90, max_depth=2, random_state=17).fit(x, y)
    state = np.random.get_state()
    try:
        np.random.seed(31)
        result = fci.random_forest_error(forest, x.shape, x, class_index=2)
    finally:
        np.random.set_state(state)
    assert result.shape == (60,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
