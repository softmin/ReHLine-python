"""Validation shared by the public estimators and solver boundary."""

from numbers import Integral, Real

import numpy as np


def positive_real(value, name, *, allow_zero=False):
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        bound = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be a finite {bound} number")


def solver_options(max_iter, tol, shrink, verbose, trace_freq):
    positive_real(tol, "tol")
    for name, value, minimum in (
        ("max_iter", max_iter, 1),
        ("trace_freq", trace_freq, 1),
        ("shrink", shrink, 0),
        ("verbose", verbose, 0),
    ):
        if not isinstance(value, Integral) or not minimum <= value <= np.iinfo(np.int32).max:
            raise ValueError(f"{name} must be an integer >= {minimum} fitting in int32")


def model_options(model):
    positive_real(model.C, "C")
    solver_options(model.max_iter, model.tol, model.shrink, model.verbose, model.trace_freq)
    if hasattr(model, "l1_ratio"):
        positive_real(model.l1_ratio, "l1_ratio", allow_zero=True)
        if model.l1_ratio >= 1:
            raise ValueError("l1_ratio must be in [0, 1)")


def named_loss_parameters(model):
    """Named-loss estimators must not silently discard a second loss definition."""
    for name in ("U", "V", "S", "T", "Tau"):
        value = getattr(model, name, None)
        if value is None:
            continue
        try:
            empty = np.asarray(value).size == 0
        except (TypeError, ValueError):
            empty = False
        if not empty:
            raise ValueError(
                f"{name} is not supported by named-loss estimators; specify loss, "
                "or use ReHLine / ReHLine_solver for manual U/V/S/T/Tau parameters"
            )


def numeric_array(value, name, *, ndim, allow_inf=False):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        value = np.asarray(value, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if np.isnan(value).any() or (not allow_inf and not np.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def sample_weights(value, n, *, allow_all_zero=False):
    if value is None:
        return np.ones(n)
    if np.ndim(value) == 0:
        value = np.full(n, value)
    value = numeric_array(value, "sample_weight", ndim=1)
    if value.shape != (n,) or (value < 0).any():
        raise ValueError(f"sample_weight must have shape ({n},) and be non-negative")
    if not allow_all_zero and not np.any(value > 0):
        raise ValueError("sample_weight must contain at least one positive weight")
    return value


def balanced_sample_weights(encoded_y, weight, n_classes):
    """Effective weights w_i * sum(w) / (K * sum(w[y == y_i])).

    Call after removing zero-weight rows and encoding the retained classes.
    This definition is independent of the installed sklearn version. Scaled
    sums and exponent arithmetic avoid overflowing class multipliers or losing
    a small class when its weights are tiny relative to another class.
    """
    class_scale = np.zeros(n_classes)
    np.maximum.at(class_scale, encoded_y, weight)
    global_scale = weight.max()
    with np.errstate(under="ignore"):
        class_mass = np.bincount(encoded_y, weights=weight / class_scale[encoded_y], minlength=n_classes)
        mass_per_class = (weight / global_scale).sum() / n_classes
    mantissa, exponent = np.frexp(weight)
    class_mantissa, class_exponent = np.frexp(class_scale)
    global_mantissa, global_exponent = np.frexp(global_scale)
    factor = (global_mantissa / class_mantissa[encoded_y]) * (mass_per_class / class_mass[encoded_y])
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            return np.ldexp(mantissa * factor, exponent + global_exponent - class_exponent[encoded_y])
    except FloatingPointError as exc:
        raise ValueError("Balanced sample weights exceed the float64 range") from exc


def quantiles(value):
    value = numeric_array(value, "quantiles", ndim=1)
    if value.size == 0 or np.any((value <= 0) | (value >= 1)):
        raise ValueError("quantiles must be a nonempty array with values in (0, 1)")
    return value
