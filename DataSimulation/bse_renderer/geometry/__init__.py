from .defect_injection import CATEGORY_IDS, apply_defect_plan, sample_defect_plan
from .surface_mesh import SurfaceState, build_surface_from_config

__all__ = [
    "SurfaceState",
    "build_surface_from_config",
    "sample_defect_plan",
    "apply_defect_plan",
    "CATEGORY_IDS",
]

