"""
line_detection_rules
ライン候補の除外判定を関数化し，症状ごとの修正をしやすくする判定群
"""

from typing import Optional

from .line_detection_config import ContourRuleConfig, PostTrackRejectConfig, SoftRecoverConfig


def is_blob_fill(fill_ratio: float, aspect: float, cfg: ContourRuleConfig) -> bool:
    return fill_ratio > cfg.blob_fill_ratio_max and aspect < cfg.blob_fill_aspect_max


def is_blob_wide(bw_ratio: float, fill_ratio: float, cfg: ContourRuleConfig) -> bool:
    return bw_ratio > cfg.blob_wide_bw_ratio_min and fill_ratio > cfg.blob_wide_fill_ratio_min


def is_row_wide(avg_row_width: float, mid_allowed_width: float, cfg: ContourRuleConfig) -> bool:
    return avg_row_width > (mid_allowed_width * cfg.row_wide_scale_max)


def is_seam_thin(avg_row_width: float, fill_ratio: float, span_ratio: float, aspect: float, cfg: ContourRuleConfig) -> bool:
    return (
        avg_row_width < cfg.seam_thin_avg_row_width_max
        and fill_ratio < cfg.seam_thin_fill_ratio_max
        and span_ratio < cfg.seam_thin_span_ratio_max
        and aspect < cfg.seam_thin_aspect_max
    )


def is_seam_horizontal(
    bw_ratio: float,
    aspect: float,
    avg_row_width: float,
    fill_ratio: float,
    cfg: ContourRuleConfig,
) -> bool:
    return (
        bw_ratio > cfg.seam_horizontal_bw_ratio_min
        and aspect < cfg.seam_horizontal_aspect_max
        and avg_row_width < cfg.seam_horizontal_avg_row_width_max
        and fill_ratio < cfg.seam_horizontal_fill_ratio_max
    )


def is_seam_snake(fill_ratio: float, avg_row_width: float, bw_ratio: float, span_ratio: float, cfg: ContourRuleConfig) -> bool:
    return (
        fill_ratio < cfg.seam_snake_fill_ratio_max
        and avg_row_width < cfg.seam_snake_avg_row_width_max
        and bw_ratio > cfg.seam_snake_bw_ratio_min
        and span_ratio > cfg.seam_snake_span_ratio_min
    )


def is_edge_blob(
    touch_left: bool,
    touch_right: bool,
    bw_ratio: float,
    fill_ratio: float,
    span_ratio: float,
    aspect: float,
    cfg: ContourRuleConfig,
) -> bool:
    return (
        (touch_left or touch_right)
        and bw_ratio > cfg.edge_blob_bw_ratio_min
        and fill_ratio > cfg.edge_blob_fill_ratio_min
        and span_ratio > cfg.edge_blob_span_ratio_min
        and aspect < cfg.edge_blob_aspect_max
    )


def is_edge_strip(
    touch_left: bool,
    touch_right: bool,
    bw_ratio: float,
    span_ratio: float,
    fill_ratio: float,
    aspect: float,
    cfg: ContourRuleConfig,
) -> bool:
    return (
        (touch_left or touch_right)
        and bw_ratio < cfg.edge_strip_bw_ratio_max
        and span_ratio > cfg.edge_strip_span_ratio_min
        and fill_ratio > cfg.edge_strip_fill_ratio_min
        and aspect > cfg.edge_strip_aspect_min
    )


def is_full_span(touch_left: bool, touch_right: bool, bw_ratio: float, cfg: ContourRuleConfig) -> bool:
    return touch_left and touch_right and bw_ratio > cfg.full_span_bw_ratio_min


def is_vertical_blob(
    touch_top: bool,
    touch_bottom: bool,
    bw_ratio: float,
    fill_ratio: float,
    aspect: float,
    cfg: ContourRuleConfig,
) -> bool:
    return (
        touch_top
        and touch_bottom
        and bw_ratio > cfg.vertical_blob_bw_ratio_min
        and fill_ratio > cfg.vertical_blob_fill_ratio_min
        and aspect < cfg.vertical_blob_aspect_max
    )


def evaluate_contour_reject_reason(
    area: float,
    span_ratio: float,
    fill_ratio: float,
    bw_ratio: float,
    avg_row_width: float,
    mid_allowed_width: float,
    aspect: float,
    touch_left: bool,
    touch_right: bool,
    touch_top: bool,
    touch_bottom: bool,
    cfg: ContourRuleConfig,
) -> Optional[str]:
    if area <= 0.0:
        return "area0"
    if area < cfg.area_min or span_ratio < cfg.span_ratio_min:
        return "small"
    if is_blob_fill(fill_ratio, aspect, cfg):
        return "blob_fill"
    if is_blob_wide(bw_ratio, fill_ratio, cfg):
        return "blob_wide"
    if is_row_wide(avg_row_width, mid_allowed_width, cfg):
        return "row_wide"
    if is_seam_thin(avg_row_width, fill_ratio, span_ratio, aspect, cfg):
        return "seam_thin"
    if is_seam_horizontal(bw_ratio, aspect, avg_row_width, fill_ratio, cfg):
        return "seam_horizontal"
    if is_seam_snake(fill_ratio, avg_row_width, bw_ratio, span_ratio, cfg):
        return "seam_snake"
    if is_edge_blob(touch_left, touch_right, bw_ratio, fill_ratio, span_ratio, aspect, cfg):
        return "edge_blob"
    if is_edge_strip(touch_left, touch_right, bw_ratio, span_ratio, fill_ratio, aspect, cfg):
        return "edge_strip"
    if is_full_span(touch_left, touch_right, bw_ratio, cfg):
        return "full_span"
    if is_vertical_blob(touch_top, touch_bottom, bw_ratio, fill_ratio, aspect, cfg):
        return "vertical_blob"
    return None


def calc_blob_penalty(width_ratio: float, span_ratio: float, cfg: ContourRuleConfig) -> float:
    if width_ratio > cfg.blob_penalty_width_ratio_min and span_ratio < cfg.blob_penalty_span_ratio_max:
        return cfg.blob_penalty_value
    return 1.0


def calc_edge_penalty(
    touch_left: bool,
    touch_right: bool,
    prev_center_x: Optional[float],
    frame_width: int,
    cfg: ContourRuleConfig,
) -> float:
    if not (touch_left or touch_right):
        return 1.0

    penalty = cfg.edge_penalty_value
    if prev_center_x is not None:
        center_offset_ratio = abs(float(prev_center_x) - (frame_width * 0.5)) / max(float(frame_width), 1.0)
        if center_offset_ratio > cfg.edge_penalty_lock_center_offset_ratio:
            penalty = cfg.edge_penalty_locked_value
    return penalty


def is_soft_recover_candidate(
    area: float,
    span_ratio: float,
    bw_ratio: float,
    fill_ratio: float,
    touch_left: bool,
    touch_right: bool,
    cfg: SoftRecoverConfig,
) -> bool:
    if area <= cfg.area_min:
        return False
    if span_ratio < cfg.span_ratio_min:
        return False
    if bw_ratio > cfg.wide_bw_ratio_max and fill_ratio > cfg.wide_fill_ratio_max:
        return False
    if touch_left and touch_right and bw_ratio > cfg.full_span_bw_ratio_min:
        return False
    return True


def evaluate_post_track_reject_reason(
    width_med: float,
    width_p90: float,
    frame_width: int,
    has_near_support: bool,
    bottom_reach_ratio: float,
    near_rows_count: int,
    near_width_med: float,
    span_ratio_rows: float,
    cfg: PostTrackRejectConfig,
) -> Optional[str]:
    is_too_wide = (
        width_med > (cfg.too_wide_med_ratio * float(frame_width))
        and width_p90 > (cfg.too_wide_p90_ratio * float(frame_width))
    )
    if is_too_wide:
        has_strong_near_support = has_near_support and (near_rows_count >= cfg.too_wide_min_near_rows_count)
        has_strong_depth_support = (
            bottom_reach_ratio >= cfg.too_wide_min_bottom_reach_ratio
            and span_ratio_rows >= cfg.too_wide_min_span_ratio_rows
        )
        if not (has_strong_near_support and has_strong_depth_support):
            return "too_wide"
    if (not has_near_support) and (width_p90 < cfg.far_thin_width_p90_max):
        return "far_thin"
    if width_med < cfg.thin_no_near_width_med_max and bottom_reach_ratio < cfg.thin_no_near_bottom_reach_max:
        return "thin_no_near"
    if (
        near_rows_count < cfg.near_weak_rows_count_max
        and near_width_med < cfg.near_weak_width_med_max
        and width_med < cfg.near_weak_global_width_med_max
    ):
        return "near_weak"
    if (
        width_med < cfg.short_span_width_med_max
        and width_p90 < cfg.short_span_width_p90_max
        and span_ratio_rows < cfg.short_span_ratio_max
    ):
        return "short_span"
    return None
