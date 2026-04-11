from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rehline")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Import from internal C++ module
from ._base import ReHLine_solver, _BaseReHLine, _make_constraint_rehline_param, _make_loss_rehline_param
from ._class import CQR_Ridge, ReHLine, plqERM_ElasticNet, plqERM_Ridge
from ._data import make_mf_dataset
from ._internal import rehline_internal, rehline_result
from ._loss import ReHLoss
from ._mf_class import plqMF_Ridge
from ._path_sol import CQR_Ridge_path_sol, plqERM_Ridge_path_sol
from ._sklearn_mixin import (
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
)

__all__ = (
    "_BaseReHLine",
    "ReHLine_solver",
    "ReHLine",
    "plqERM_Ridge",
    "CQR_Ridge",
    "plqERM_ElasticNet",
    "plqMF_Ridge",
    "CQR_Ridge_path_sol",
    "plqERM_Ridge_path_sol",
    "plq_Ridge_Classifier",
    "plq_Ridge_Regressor",
    "plq_ElasticNet_Classifier",
    "plq_ElasticNet_Regressor",
    "_make_loss_rehline_param",
    "_make_constraint_rehline_param",
    "make_mf_dataset",
)
