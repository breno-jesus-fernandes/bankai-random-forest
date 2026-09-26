from sklearn.utils.estimator_checks import check_estimator

from bankai_random_forest import BankaiRandomForestClassifier


def test_estimator_passes_sklearn_common_checks():
    check_estimator(BankaiRandomForestClassifier(), on_fail="raise", on_skip=None)
