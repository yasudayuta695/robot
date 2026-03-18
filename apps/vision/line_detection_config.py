"""
line_detection_config
ライン検出の判定しきい値を集約し，調整しやすくする設定定義
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ContourRuleConfig:
    """輪郭候補の除外判定に使うしきい値群"""

    area_min: float = 120.0
    span_ratio_min: float = 0.08

    blob_fill_ratio_max: float = 0.72
    blob_fill_aspect_max: float = 2.0

    blob_wide_bw_ratio_min: float = 0.62
    blob_wide_fill_ratio_min: float = 0.35
    row_wide_scale_max: float = 1.85

    seam_thin_avg_row_width_max: float = 10.0
    seam_thin_fill_ratio_max: float = 0.24
    seam_thin_span_ratio_max: float = 0.32
    seam_thin_aspect_max: float = 0.95

    seam_horizontal_bw_ratio_min: float = 0.30
    seam_horizontal_aspect_max: float = 0.42
    seam_horizontal_avg_row_width_max: float = 16.0
    seam_horizontal_fill_ratio_max: float = 0.34

    seam_snake_fill_ratio_max: float = 0.12
    seam_snake_avg_row_width_max: float = 14.0
    seam_snake_bw_ratio_min: float = 0.34
    seam_snake_span_ratio_min: float = 0.42

    edge_blob_bw_ratio_min: float = 0.70
    edge_blob_fill_ratio_min: float = 0.26
    edge_blob_span_ratio_min: float = 0.30
    edge_blob_aspect_max: float = 1.85

    edge_strip_bw_ratio_max: float = 0.05
    edge_strip_span_ratio_min: float = 0.65
    edge_strip_fill_ratio_min: float = 0.30
    edge_strip_aspect_min: float = 3.4

    full_span_bw_ratio_min: float = 0.25

    vertical_blob_bw_ratio_min: float = 0.34
    vertical_blob_fill_ratio_min: float = 0.30
    vertical_blob_aspect_max: float = 2.0

    blob_penalty_width_ratio_min: float = 0.82
    blob_penalty_span_ratio_max: float = 0.55
    blob_penalty_value: float = 0.65

    edge_penalty_value: float = 0.72
    edge_penalty_locked_value: float = 0.55
    edge_penalty_lock_center_offset_ratio: float = 0.30


@dataclass
class SoftRecoverConfig:
    """厳格判定が全滅した場合の救済判定に使うしきい値群"""

    area_min: float = 80.0
    span_ratio_min: float = 0.06
    wide_bw_ratio_max: float = 0.90
    wide_fill_ratio_max: float = 0.40
    full_span_bw_ratio_min: float = 0.70


@dataclass
class PostTrackRejectConfig:
    """中心線追跡後の最終除外判定に使うしきい値群"""

    final_contour_area_min: float = 220.0
    too_wide_med_ratio: float = 0.18
    too_wide_p90_ratio: float = 0.28

    far_thin_width_p90_max: float = 16.0
    thin_no_near_width_med_max: float = 9.0
    thin_no_near_bottom_reach_max: float = 0.72

    near_weak_rows_count_max: int = 14
    near_weak_width_med_max: float = 14.0
    near_weak_global_width_med_max: float = 11.0

    short_span_width_med_max: float = 8.0
    short_span_width_p90_max: float = 13.0
    short_span_ratio_max: float = 0.35


@dataclass
class LineDetectionConfig:
    """ライン検出全体の判定設定"""

    contour: ContourRuleConfig = field(default_factory=ContourRuleConfig)
    soft_recover: SoftRecoverConfig = field(default_factory=SoftRecoverConfig)
    post_track: PostTrackRejectConfig = field(default_factory=PostTrackRejectConfig)


DEFAULT_LINE_DETECTION_PROFILE = "default"
PANEL_SEAM_LINE_DETECTION_PROFILE = "panel_seam"
GLARE_LINE_DETECTION_PROFILE = "glare"
PANEL_SEAM_GLARE_LINE_DETECTION_PROFILE = "panel_seam_glare"


def _apply_panel_seam_tuning(cfg: LineDetectionConfig) -> None:
    cfg.contour.seam_thin_avg_row_width_max = 12.0
    cfg.contour.seam_thin_fill_ratio_max = 0.30
    cfg.contour.seam_thin_span_ratio_max = 0.40
    cfg.contour.seam_horizontal_bw_ratio_min = 0.26
    cfg.contour.seam_horizontal_aspect_max = 0.50
    cfg.contour.seam_horizontal_fill_ratio_max = 0.40
    cfg.contour.seam_snake_fill_ratio_max = 0.16
    cfg.contour.seam_snake_avg_row_width_max = 16.0


def _apply_glare_tuning(cfg: LineDetectionConfig) -> None:
    cfg.post_track.final_contour_area_min = 180.0
    cfg.post_track.far_thin_width_p90_max = 13.0
    cfg.post_track.thin_no_near_width_med_max = 7.5
    cfg.post_track.thin_no_near_bottom_reach_max = 0.66
    cfg.post_track.near_weak_rows_count_max = 10
    cfg.post_track.near_weak_width_med_max = 11.0
    cfg.post_track.near_weak_global_width_med_max = 9.0
    cfg.post_track.short_span_width_med_max = 7.0
    cfg.post_track.short_span_width_p90_max = 11.5
    cfg.post_track.short_span_ratio_max = 0.26


def build_line_detection_profile(profile_name: str = DEFAULT_LINE_DETECTION_PROFILE) -> LineDetectionConfig:
    profile = str(profile_name).strip().lower()

    cfg = LineDetectionConfig()
    if profile == PANEL_SEAM_LINE_DETECTION_PROFILE:
        # 目地誤検出を抑える設定，細線と横長線の除外をやや強化
        _apply_panel_seam_tuning(cfg)
        return cfg

    if profile == GLARE_LINE_DETECTION_PROFILE:
        # 白飛び環境で線が欠けるケース向け，post rejectを緩和
        _apply_glare_tuning(cfg)
        return cfg

    if profile in {DEFAULT_LINE_DETECTION_PROFILE, PANEL_SEAM_GLARE_LINE_DETECTION_PROFILE}:
        # 目地抑制と白飛び緩和を同時に適用
        _apply_panel_seam_tuning(cfg)
        _apply_glare_tuning(cfg)
        return cfg

    return cfg


def list_line_detection_profiles() -> Dict[str, str]:
    return {
        DEFAULT_LINE_DETECTION_PROFILE: "既定設定（目地抑制+白飛び緩和）",
        PANEL_SEAM_LINE_DETECTION_PROFILE: "パネル目地抑制",
        GLARE_LINE_DETECTION_PROFILE: "白飛び緩和",
        PANEL_SEAM_GLARE_LINE_DETECTION_PROFILE: "目地抑制+白飛び緩和",
    }
