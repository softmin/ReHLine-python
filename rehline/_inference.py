"""Independent prediction snapshots with no training loss matrices or duals.

Construct these through a fitted estimator's to_inference() method. Prediction
uses the estimator's existing implementations, including bounded OvO voting.
"""

from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._class import CQR_Ridge
from ._sklearn_mixin import _ReHLineClassifier, _SklearnReHLine

_FITTED_FIELDS = (
    "coef_",
    "intercept_",
    "n_features_in_",
    "feature_names_in_",
    "n_iter_",
    "objective_",
    "dual_objective_",
    "dual_gap_",
    "constraint_violation_",
    "scaled_constraint_violation_",
    "kkt_residual_",
    "converged_",
)


class _InferenceBase(BaseEstimator):
    def fit(self, *args, **kwargs):
        raise TypeError("This inference snapshot cannot fit or warm-start; use the original estimator to refit")


class _LinearInference(_InferenceBase):
    _decision_function = _SklearnReHLine._decision_function
    predict = _SklearnReHLine.predict


class _RegressorInference(RegressorMixin, _LinearInference):
    pass


class _ClassifierInference(ClassifierMixin, _LinearInference):
    def __init__(self, decision_function_shape="ovr"):
        self.decision_function_shape = decision_function_shape

    _class_scores = _SklearnReHLine._class_scores
    _ovo_class_scores = _SklearnReHLine._ovo_class_scores
    _validate_decision_function_shape = _ReHLineClassifier._validate_decision_function_shape
    decision_function = _ReHLineClassifier.decision_function


class _CQRInference(_InferenceBase):
    predict = CQR_Ridge.predict


def _copy_fields(model, snapshot, fields):
    for name in fields:
        if hasattr(model, name):
            setattr(snapshot, name, deepcopy(getattr(model, name)))
    snapshot.source_estimator_ = type(model).__name__
    return snapshot


def _cqr_snapshot(model):
    check_is_fitted(model)
    return _copy_fields(model, _CQRInference(), (*_FITTED_FIELDS, "quantiles_"))


def _sklearn_snapshot(model):
    check_is_fitted(model, ["coef_", "intercept_"])
    classifier = isinstance(model, ClassifierMixin)
    if classifier:
        model._validate_decision_function_shape()
        snapshot = _ClassifierInference(model.decision_function_shape)
    else:
        snapshot = _RegressorInference()
    _copy_fields(model, snapshot, _FITTED_FIELDS)
    if classifier:
        _copy_fields(model, snapshot, ("classes_", "multi_class_", "_label_encoder"))
        if len(model.classes_) > 2 and model.multi_class_ == "ovo":
            # Inference only needs the pair ordering; coefficient rows are views
            # into the snapshot's own array, never into the training estimator.
            snapshot.estimators_ = [
                (snapshot.coef_[k], snapshot.intercept_[k], a, b) for k, (_, _, a, b) in enumerate(model.estimators_)
            ]
    return snapshot
