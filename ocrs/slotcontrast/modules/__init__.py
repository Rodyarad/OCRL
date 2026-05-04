from . import timm
from .decoders import build as build_decoder
from .dynamics import build as build_dynamics_predictor
from .encoders import build as build_encoder
from .groupers import build as build_grouper
from .initializers import build as build_initializer
from .networks import build as build_network
from .utils import Resizer, SoftToHardMask
from .utils import build as build_utils
from .utils import build_module, build_torch_function, build_torch_module
from .video import LatentProcessor, MapOverTime, ScanOverTime
from .video import build as build_video

__all__ = [
    "build_decoder",
    "build_dynamics_predictor",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "dynamics_predictors": build_dynamics_predictor,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
