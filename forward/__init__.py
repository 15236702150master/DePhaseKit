"""正演与厚度反演子包。

论文的方法学卖点是"两条相互独立的正演路径互相校验"，本子包是这条方法学的
可运行实现：

    from forward import analytic_thickness, grid_search_thickness, cross_validate

    cross_validate()          # 复算论文的 1.13% 双路径一致性
    grid_search_thickness(...)  # 单组走时差 -> 厚度（TauP 精确射线追踪）
    analytic_thickness(...)     # 平层解析公式

速度默认值见 forward.constants（Vp = 6.0 km/s，Allen 1966 折射实测）。
"""
from .constants import (
    DEFAULT_CRUST_RHO,
    DEFAULT_CRUST_VP,
    DEFAULT_CRUST_VS,
    DEFAULT_H_MAX,
    DEFAULT_H_MIN,
    DEFAULT_H_STEP,
    DEFAULT_PICK_UNCERTAINTY,
    DEFAULT_VP_UNCERTAINTY,
)
from .taup_moho import (
    GridResult,
    analytic_sensitivity,
    analytic_thickness,
    cross_validate,
    get_moho_model,
    grid_search_thickness,
    reference_ray_parameter,
    taup_differential,
    thickness_uncertainty,
)

__all__ = [
    "DEFAULT_CRUST_RHO",
    "DEFAULT_CRUST_VP",
    "DEFAULT_CRUST_VS",
    "DEFAULT_H_MAX",
    "DEFAULT_H_MIN",
    "DEFAULT_H_STEP",
    "DEFAULT_PICK_UNCERTAINTY",
    "DEFAULT_VP_UNCERTAINTY",
    "GridResult",
    "analytic_sensitivity",
    "analytic_thickness",
    "cross_validate",
    "get_moho_model",
    "grid_search_thickness",
    "reference_ray_parameter",
    "taup_differential",
    "thickness_uncertainty",
]
