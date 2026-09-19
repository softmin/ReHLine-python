"""Identify every GridSearch estimator fit and its native binary subproblems.

Instrumentation is confined to the benchmark process and restored on exit.
The outer GridSearch is serial; inner multiclass tasks may run in threads.
"""

import json
from contextlib import contextmanager
from itertools import combinations
from threading import local

import numpy as np
from joblib import parallel_config
from sklearn.base import is_classifier
from sklearn.utils.class_weight import compute_class_weight


class FitRecords(list):
    def __init__(self):
        super().__init__()
        self.manifest = []


def subproblems(classes, strategy):
    if strategy == "regression":
        return [("regression", [])]
    if len(classes) < 2:
        raise ValueError("Objective manifest requires at least two retained classes")
    if len(classes) == 2:
        return [("binary", list(classes))]
    if strategy == "ovr":
        return [("ovr", [c]) for c in classes]
    if strategy == "ovo":
        return [("ovo", list(pair)) for pair in combinations(classes, 2)]
    raise ValueError("Unknown strategy in objective manifest")


def record_key(fit_id, kind, classes):
    return json.dumps([fit_id, kind, classes], ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def expected_keys(manifest, n_candidates, cv):
    """Derive counts from retained classes, never from recorded native calls."""
    if len(manifest) != n_candidates * cv + 1:
        raise ValueError("Missing candidate or fold estimator fits in objective manifest")
    keys = []
    for index, entry in enumerate(manifest):
        candidate, fold = divmod(index, cv) if index < n_candidates * cv else ("best", "refit")
        if (entry["fit_id"], entry["candidate"], entry["fold"]) != (index, candidate, fold):
            raise ValueError("Wrong candidate or fold identity in objective manifest")
        classes = entry["classes"]
        if len(set(json.dumps(c) for c in classes)) != len(classes):
            raise ValueError("Duplicate classes in objective manifest")
        keys.extend(record_key(index, kind, labels) for kind, labels in subproblems(classes, entry["strategy"]))
    return keys


def keyed_records(records, expected):
    keys = [record["fit_key"] for record in records]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("Missing, duplicate or unexpected binary subproblem in objective audit")
    return dict(zip(keys, records))


def retained_classes(model, y, sample_weight):
    """Independently mirror only the documented reference-row selection."""
    y = np.asarray(y)
    weight = np.ones(len(y)) if sample_weight is None else np.broadcast_to(sample_weight, (len(y),))
    y = y[weight > 0]
    classes = np.unique(y)
    if model.class_weight is not None:
        weights = compute_class_weight(model.class_weight, classes=classes, y=y)
        classes = classes[weights > 0]
    return classes.tolist()


@contextmanager
def track_fits(records, *, cv, n_candidates, audit):
    from rehline._class import CQR_Ridge
    from rehline._sklearn_mixin import _SklearnReHLine

    context = local()
    originals = []
    marker = "_benchmark_fit_context"

    @contextmanager
    def current(value):
        old = getattr(context, "value", None)
        context.value = value
        try:
            yield
        finally:
            context.value = old

    def outer(original):
        def fit(model, X, y, *args, **kwargs):
            index = len(records.manifest)
            candidate, fold = divmod(index, cv) if index < n_candidates * cv else ("best", "refit")
            classifier = is_classifier(model)
            strategy = (model.multi_class or "ovr") if classifier else "regression"
            entry = dict(fit_id=index, candidate=candidate, fold=fold, strategy=strategy, classes=[])
            if audit and classifier:
                entry["classes"] = retained_classes(model, y, kwargs.get("sample_weight", args[0] if args else None))
            records.manifest.append(entry)
            old_marker = getattr(model, marker, None)
            info = dict(fit_id=index, objective_scale=1 - getattr(model, "l1_ratio", 0))
            setattr(model, marker, info)
            try:
                with current(dict(info, kind="regression", classes=[])):
                    result = original(model, X, y, *args, **kwargs)
                if classifier and not audit:
                    entry["classes"] = model.classes_.tolist()
                return result
            finally:
                if old_marker is None:
                    model.__dict__.pop(marker, None)
                else:
                    setattr(model, marker, old_marker)

        return fit

    def task(original):
        def fit(model, X, y, weight, key, rows, previous):
            info = getattr(model, marker)
            labels = [label.item() if isinstance(label, np.generic) else label for label in key]
            with current(dict(info, kind=model.multi_class_, classes=labels)):
                return original(model, X, y, weight, key, rows, previous)

        return fit

    def binary(original):
        def fit(model, X, y, weight, previous=None):
            info = getattr(model, marker)
            if is_classifier(model) and len(model.classes_) == 2:
                with current(dict(info, kind="binary", classes=model.classes_.tolist())):
                    return original(model, X, y, weight, previous)
            return original(model, X, y, weight, previous)

        return fit

    try:
        for cls, name, wrapper in (
            (_SklearnReHLine, "fit", outer),
            (CQR_Ridge, "fit", outer),
            (_SklearnReHLine, "_fit_multiclass_task", task),
            (_SklearnReHLine, "_fit_model", binary),
        ):
            original = getattr(cls, name)
            originals.append((cls, name, original))
            setattr(cls, name, wrapper(original))
        # Process workers would not inherit this process's instrumentation.
        with parallel_config(backend="threading"):
            yield context
        expected = expected_keys(records.manifest, n_candidates, cv)
        keyed_records(records, expected)
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)
