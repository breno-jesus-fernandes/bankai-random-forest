from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


def test_default_constructor_matches_sklearn_random_forest_classifier():
    from bankai_random_forest import BankaiRandomForestClassifier

    estimator = BankaiRandomForestClassifier()

    assert isinstance(estimator, ClassifierMixin)
    assert isinstance(estimator, BaseEstimator)
    assert estimator.get_params(deep=False) == RandomForestClassifier().get_params(
        deep=False
    )
