import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import skia
import uharfbuzz as hb

from core.text.text_processing import (
    STYLE_PATTERN,
    find_optimal_breaks_dp,
    parse_styled_segments,
    reflow_diamond,
    tokenize_styled_text,
    try_hyphenate_word,
    typeR_diamond_break,
)
from utils.exceptions import RenderingError
from utils.logging import log_message

# Epsilon to guard rounding when converting from HarfBuzz 26.6 fixed-point.
VISUAL_WIDTH_EPSILON = 0.0

def _contains_rtl_script(text: str) -> bool:
    """Check if *text* contains characters from an RTL script (Arabic, Hebrew, etc.)."""
    for ch in text:
        cp = ord(ch)
        if 0x0590 <= cp <= 0x05FF:    # Hebrew
            return True
        if 0x0600 <= cp <= 0x06FF:    # Arabic
            return True
        if 0x0700 <= cp <= 0x074F:    # Syriac
            return True
        if 0xFB1D <= cp <= 0xFDFF:    # Hebrew / Arabic presentation forms
            return True
        if 0xFE70 <= cp <= 0xFEFF:    # Arabic presentation forms-B
            return True
        if 0x10800 <= cp <= 0x10FFF:  # Other RTL scripts
            return True
    return False


def shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition], str]:
    """Shapes a line of text with HarfBuzz.

    Returns:
        Tuple of (glyph_infos, glyph_positions, direction) where direction
        is "ltr" or "rtl" as detected by HarfBuzz.

    Raises:
        RenderingError: If HarfBuzz shaping fails
    """
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    # Explicitly set direction for RTL scripts (Arabic, Hebrew, etc.)
    # to avoid auto-detection issues with mixed content (e.g. Arabic + numbers).
    if _contains_rtl_script(text_line):
        hb_buffer.direction = "rtl"
    else:
        hb_buffer.guess_segment_properties()
    direction = str(hb_buffer.direction)
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions, direction
    except Exception as e:
        log_message(f"HarfBuzz shaping failed: {e}", always_print=True)
        raise RenderingError("HarfBuzz text shaping failed") from e


def calculate_line_width(positions: List[hb.GlyphPosition]) -> float:
    """Calculate visual width using advances and first/last x_offset."""
    if not positions:
        return 0.0
    HB_26_6_SCALE_FACTOR = 64.0

    total_advance_fixed = sum(pos.x_advance for pos in positions)
    first_offset_fixed = positions[0].x_offset
    last_offset_fixed = positions[-1].x_offset

    visual_width_fixed = total_advance_fixed + (last_offset_fixed - first_offset_fixed)
    visual_width = float(visual_width_fixed / HB_26_6_SCALE_FACTOR)
    return visual_width + VISUAL_WIDTH_EPSILON


def calculate_styled_line_width(
    line_with_markers: str,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
) -> float:
    """Calculate the width of a line that may contain style markers.

    Uses the appropriate HarfBuzz faces per style segment, falling back to
    the 'regular' face if a style-specific face is missing.
    """
    if not line_with_markers:
        return 0.0

    segments = parse_styled_segments(line_with_markers)
    if not segments:
        return 0.0

    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return 0.0

    total_advance_fixed_all = 0
    first_offset_fixed_global: Optional[int] = None
    last_offset_fixed_global: Optional[int] = None

    for segment_text, style_name in segments:
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face

        hb_font_segment = hb.Font(hb_face_to_use)
        hb_font_segment.ptem = float(font_size)
        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font_segment.scale = (hb_scale, hb_scale)

        _, positions, _ = shape_line(segment_text, hb_font_segment, features)
        if not positions:
            continue

        total_advance_fixed_all += sum(pos.x_advance for pos in positions)
        if first_offset_fixed_global is None:
            first_offset_fixed_global = positions[0].x_offset
        last_offset_fixed_global = positions[-1].x_offset

    if total_advance_fixed_all == 0 and first_offset_fixed_global is None:
        return 0.0

    HB_26_6_SCALE_FACTOR = 64.0
    offset_delta_fixed = 0
    if first_offset_fixed_global is not None and last_offset_fixed_global is not None:
        offset_delta_fixed = last_offset_fixed_global - first_offset_fixed_global

    visual_width_fixed_all = total_advance_fixed_all + offset_delta_fixed
    visual_width_all = float(visual_width_fixed_all / HB_26_6_SCALE_FACTOR)
    return visual_width_all + VISUAL_WIDTH_EPSILON


def check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphen_penalty: float,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
    detach_trailing_ellipsis: bool = True,
) -> Optional[Dict]:
    """Check if text fits within the given dimensions at the specified font size.

    Args:
        font_size: Font size to test
        text: Text to wrap and measure
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for shaping
        regular_typeface: Skia typeface for metrics
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before scaling
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(regular_typeface, font_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (
                -metrics.fAscent + metrics.fDescent + metrics.fLeading
            ) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = font_size * 1.2 * line_spacing_mult
        except Exception as e:
            if verbose:
                log_message(
                    f"Font metrics unavailable at size {font_size}: {e}",
                    verbose=verbose,
                )
            single_line_height = font_size * 1.2 * line_spacing_mult

        # Respect explicit newlines as hard line breaks (e.g., for vertical stacking)
        if "\n" in text:
            explicit_lines = text.split("\n")
            current_max_line_width = 0.0
            lines_data_at_size = []
            for line_text in explicit_lines:
                width = calculate_styled_line_width(
                    line_text, font_size, loaded_hb_faces, features_to_enable
                )
                lines_data_at_size.append(
                    {"text_with_markers": line_text, "width": width}
                )
                current_max_line_width = max(current_max_line_width, width)

            total_block_height = (-metrics.fAscent + metrics.fDescent) + (
                len(explicit_lines) - 1
            ) * single_line_height

            if (
                current_max_line_width <= max_render_width
                and total_block_height <= max_render_height
            ):
                return {
                    "lines": lines_data_at_size,
                    "metrics": metrics,
                    "max_line_width": current_max_line_width,
                    "line_height": single_line_height,
                }
            return None

        tokens: List[Tuple[str, bool]] = tokenize_styled_text(
            text, detach_trailing_ellipsis
        )
        augmented_tokens: List[str] = []

        if hyphenate_before_scaling:
            for token_text, is_styled in tokens:
                marker = ""
                core_text = token_text

                if is_styled:
                    styled_match = STYLE_PATTERN.match(token_text)
                    if not styled_match:
                        augmented_tokens.append(token_text)
                        continue
                    marker = styled_match.group(1)
                    core_text = styled_match.group(2)

                match = re.match(r"^(\W*)([\w\-]+)(\W*)$", core_text)
                if match:
                    core_word_length = len(match.group(2))
                else:
                    core_word_length = len(core_text)

                if core_word_length > hyphenation_min_word_length:
                    word_width = calculate_styled_line_width(
                        token_text, font_size, loaded_hb_faces, features_to_enable
                    )

                    if word_width > max_render_width:

                        def wrap_part(part: str) -> str:
                            return f"{marker}{part}{marker}" if marker else part

                        def width_test_func(part: str) -> bool:
                            wrapped = wrap_part(part)
                            w = calculate_styled_line_width(
                                wrapped, font_size, loaded_hb_faces, features_to_enable
                            )
                            return w <= max_render_width

                        split_parts = try_hyphenate_word(
                            core_text, hyphenation_min_word_length, width_test_func
                        )
                        if split_parts:
                            augmented_tokens.extend(wrap_part(p) for p in split_parts)
                        else:
                            augmented_tokens.append(token_text)
                    else:
                        augmented_tokens.append(token_text)
                else:
                    augmented_tokens.append(token_text)
        else:
            augmented_tokens = [t for t, _ in tokens]

        try:
            GLUE_TRAILING_PUNCT_RE = re.compile(r"^[,.;:!?…]+$")
            GLUE_CLOSERS_RE = re.compile(r"^[\)\]\}\u2019\u201D\'\"]+$")

            def _glue_trailing_punctuation(
                tokens_list: List[str], _detach: bool = True
            ) -> List[str]:
                glued: List[str] = []
                for tok in tokens_list:
                    match = STYLE_PATTERN.match(tok)
                    content = match.group(2) if match else tok

                    # Skip gluing for disconnected ellipsis to allow wrapping
                    if _detach and re.match(
                        r"^(\.{2,})[\)\]\}\u2019\u201D\'\"]*$", content
                    ):
                        glued.append(tok)
                        continue

                    if glued and (
                        GLUE_TRAILING_PUNCT_RE.match(content)
                        or GLUE_CLOSERS_RE.match(content)
                    ):
                        glued[-1] = glued[-1] + tok
                    else:
                        glued.append(tok)
                return glued

            augmented_tokens = _glue_trailing_punctuation(
                augmented_tokens, detach_trailing_ellipsis
            )
        except Exception:
            pass

        def word_width_func(word: str) -> float:
            if word_width_cache is not None:
                cached_key = (word, font_size)
                if cached_key in word_width_cache:
                    return word_width_cache[cached_key]

            width_val = calculate_styled_line_width(
                word, font_size, loaded_hb_faces, features_to_enable
            )

            if word_width_cache is not None:
                word_width_cache[(word, font_size)] = width_val

            return width_val

        space_width = calculate_styled_line_width(
            " ", font_size, loaded_hb_faces, features_to_enable
        )

        wrapped_lines_text = find_optimal_breaks_dp(
            augmented_tokens,
            max_render_width,
            word_width_func,
            space_width,
            badness_exponent,
            hyphen_penalty,
            detach_trailing_ellipsis,
        )

        if not wrapped_lines_text:
            if verbose:
                log_message(f"Size {font_size}: line break DP failed", verbose=verbose)
            return None

        # Reshape lines to mirror oval bubble: diamond pattern.
        # TypeR-inspired DP distributes words across diamond-shaped targets.
        # Falls back to post-hoc reflow_diamond if TypeR returns None.
        if len(wrapped_lines_text) >= 3:
            typeR_succeeded = False
            try:
                typeR_result = typeR_diamond_break(
                    augmented_tokens,
                    max_render_width,
                    max_render_height,
                    word_width_func,
                    space_width,
                    detach_trailing_ellipsis,
                    font_size,
                )
                if typeR_result is not None and len(typeR_result) >= 3:
                    wrapped_lines_text = typeR_result
                    typeR_succeeded = True
            except Exception:
                pass
            if not typeR_succeeded:
                try:
                    reflowed = reflow_diamond(
                        wrapped_lines_text,
                        max_render_width,
                        word_width_func,
                        space_width,
                    )
                    if reflowed and len(reflowed) == len(wrapped_lines_text):
                        wrapped_lines_text = reflowed
                except Exception:
                    pass

        current_max_line_width = 0
        lines_data_at_size = []
        for line_text_with_markers in wrapped_lines_text:
            width = calculate_styled_line_width(
                line_text_with_markers, font_size, loaded_hb_faces, features_to_enable
            )
            lines_data_at_size.append(
                {"text_with_markers": line_text_with_markers, "width": width}
            )
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (
            len(wrapped_lines_text) - 1
        ) * single_line_height

        if verbose:
            log_message(
                f"Size {font_size}: {current_max_line_width:.0f}x{total_block_height:.0f} "
                f"(max {max_render_width:.0f}x{max_render_height:.0f})",
                verbose=verbose,
            )

        if (
            current_max_line_width <= max_render_width
            and total_block_height <= max_render_height
        ):
            if verbose:
                log_message(f"Size {font_size} fits", verbose=verbose)
            return {
                "lines": lines_data_at_size,
                "metrics": metrics,
                "max_line_width": current_max_line_width,
                "line_height": single_line_height,
            }

        return None

    except Exception as e:
        log_message(f"Fit check EXCEPTION at size {font_size}: {e}", always_print=True)
        return None


def _check_collision(
    lines_data: List[Dict],
    box_top_left: Tuple[int, int],
    cleaned_mask: np.ndarray,
    line_height: float,
    render_size: Tuple[float, float],
) -> bool:
    """
    Check if any text pixel overlaps with background (0) in mask.

    Args:
        lines_data: List of dictionaries containing line width and text.
        box_top_left: (x, y) coordinates of the bounding box top-left corner.
        cleaned_mask: Binary mask of the bubble (0=background, 255=bubble).
        line_height: Height of a single line of text.
        render_size: (width, height) of the render box.

    Returns:
        True if collision detected, False otherwise.
    """
    box_x, box_y = box_top_left
    mask_h, mask_w = cleaned_mask.shape
    max_w, max_h = render_size

    total_text_height = len(lines_data) * line_height
    start_y = box_y + (max_h - total_text_height) / 2

    current_y = start_y
    for line in lines_data:
        line_w = line["width"]
        line_x = box_x + (max_w - line_w) / 2

        y1, y2 = int(current_y), int(current_y + line_height)
        x1, x2 = int(line_x), int(line_x + line_w)

        points_to_check = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

        for px, py in points_to_check:
            px = max(0, min(px, mask_w - 1))
            py = max(0, min(py, mask_h - 1))

            if cleaned_mask[py, px] == 0:
                return True

        current_y += line_height

    return False


def _compute_fill_ratio(
    n_lines: int, line_height: float, max_line_width: float,
    box_w: float, box_h: float
) -> float:
    """
    Return vertical_fill / horizontal_fill.
    A value of 1.0 means the text block is proportional to the bubble.
    Values < 1 mean vertical space is mostly empty (text too wide, not tall).
    Target: ratio >= MIN_FILL_RATIO so margins are roughly equal on all sides.
    """
    h_fill = max_line_width / max(box_w, 1.0)
    v_fill = (n_lines * line_height) / max(box_h, 1.0)
    return v_fill / max(h_fill, 0.01)


# Minimum fill-ratio threshold.
# 0.90 means vertical fill must be at least 90% of horizontal fill.
# Higher values produce more balanced margins; lower values prioritise
# larger text at the cost of uneven top/bottom vs left/right spacing.
_MIN_FILL_RATIO = 0.90


def find_optimal_layout(
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    min_font_size: int = 8,
    max_font_size: int = 16,
    line_spacing_mult: float = 1.0,
    hyphenate_before_scaling: bool = True,
    hyphen_penalty: float = 1000.0,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 2.0,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    cleaned_mask: Optional[np.ndarray] = None,
    box_top_left: Optional[Tuple[int, int]] = None,
    detach_trailing_ellipsis: bool = True,
) -> Dict:
    """Find the optimal font size and layout for text within given dimensions.

    Uses binary search to find the largest font size that fits.

    Args:
        text: Text to layout
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for the regular font
        regular_typeface: Skia typeface for the regular font
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        min_font_size: Minimum font size to try
        max_font_size: Maximum font size to try
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before reducing font size
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        verbose: Whether to print detailed logs
        bubble_id: Optional identifier for the bubble (for logging purposes)
        cleaned_mask: Optional binary mask of the bubble for collision detection
        box_top_left: Optional (x, y) coordinates of the bounding box top-left corner

    Returns:
        Dictionary containing layout data (font_size, lines, metrics, etc.)

    Raises:
        RenderingError: If text doesn't fit at minimum font size or layout fails
    """
    # Preserve explicit newlines if present (e.g., vertical stacking),
    # otherwise collapse whitespace for normal paragraph layout
    if "\n" in text or "\r" in text:
        clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    else:
        clean_text = " ".join(text.split())
    if not clean_text:
        raise RenderingError("Empty text cannot be laid out")

    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")
    best_fit_line_height = 0.0

    word_width_cache: Dict[Tuple[str, int], float] = {}

    low = min_font_size
    high = max_font_size

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break

        log_message(f"Testing size {mid}", verbose=verbose)

        succeeded_at_current_size = False
        current_width_attempt = max_render_width
        max_squeezes = 3 if cleaned_mask is not None else 1

        for _ in range(max_squeezes):
            fit_data = check_fit(
                mid,
                clean_text,
                current_width_attempt,
                max_render_height,
                regular_hb_face,
                regular_typeface,
                loaded_hb_faces,
                features_to_enable,
                line_spacing_mult,
                hyphenate_before_scaling,
                hyphen_penalty,
                hyphenation_min_word_length,
                badness_exponent,
                word_width_cache,
                verbose,
                detach_trailing_ellipsis,
            )

            if fit_data is None:
                # Squeezing narrower won't help (only makes it taller)
                break

            if cleaned_mask is not None and box_top_left is not None:
                has_collision = _check_collision(
                    fit_data["lines"],
                    box_top_left,
                    cleaned_mask,
                    fit_data["line_height"],
                    (current_width_attempt, max_render_height),
                )

                if not has_collision:
                    best_fit_size = mid
                    best_fit_lines_data = fit_data["lines"]
                    best_fit_metrics = fit_data["metrics"]
                    best_fit_max_line_width = fit_data["max_line_width"]
                    best_fit_line_height = fit_data["line_height"]

                    succeeded_at_current_size = True
                    break
                else:
                    if verbose:
                        log_message(
                            f"Collision at size {mid} width {current_width_attempt:.0f}, squeezing...",
                            verbose=verbose,
                        )
                    current_width_attempt *= 0.90
                    continue
            else:
                best_fit_size = mid
                best_fit_lines_data = fit_data["lines"]
                best_fit_metrics = fit_data["metrics"]
                best_fit_max_line_width = fit_data["max_line_width"]
                best_fit_line_height = fit_data["line_height"]
                succeeded_at_current_size = True
                break

        if succeeded_at_current_size:
            low = mid + 1
        else:
            high = mid - 1

    # ── Fallback: try font sizes below min_font_size ──────────────────────────
    if best_fit_size == -1:
        HARD_MIN = 4
        fallback_size = min_font_size - 1
        while fallback_size >= HARD_MIN and best_fit_size == -1:
            log_message(f"Fallback testing size {fallback_size}", verbose=verbose)
            current_width_attempt = max_render_width
            max_squeezes = 3 if cleaned_mask is not None else 1
            for _ in range(max_squeezes):
                fit_data = check_fit(
                    fallback_size, clean_text, current_width_attempt, max_render_height,
                    regular_hb_face, regular_typeface, loaded_hb_faces, features_to_enable,
                    line_spacing_mult, hyphenate_before_scaling, hyphen_penalty,
                    hyphenation_min_word_length, badness_exponent, word_width_cache,
                    verbose, detach_trailing_ellipsis,
                )
                if fit_data is None:
                    break
                if cleaned_mask is not None and box_top_left is not None:
                    has_collision = _check_collision(
                        fit_data["lines"], box_top_left, cleaned_mask,
                        fit_data["line_height"], (current_width_attempt, max_render_height),
                    )
                    if not has_collision:
                        best_fit_size = fallback_size
                        best_fit_lines_data = fit_data["lines"]
                        best_fit_metrics = fit_data["metrics"]
                        best_fit_max_line_width = fit_data["max_line_width"]
                        best_fit_line_height = fit_data["line_height"]
                        break
                    else:
                        current_width_attempt *= 0.90
                        continue
                else:
                    best_fit_size = fallback_size
                    best_fit_lines_data = fit_data["lines"]
                    best_fit_metrics = fit_data["metrics"]
                    best_fit_max_line_width = fit_data["max_line_width"]
                    best_fit_line_height = fit_data["line_height"]
                    break
            fallback_size -= 1

    if best_fit_size == -1:
        log_message(
            f"Text too large for bubble at min size {min_font_size}: '{clean_text[:30]}'",
            always_print=True,
        )
        raise RenderingError(
            f"Text too large for bubble at minimum font size {min_font_size}"
        )

    # ── Post-optimisation: fill-ratio balancing ──────────────────────────────
    # Goal: make vertical fill ≈ horizontal fill so margins are equal on all
    # sides of the bubble (top/bottom ≈ left/right).
    #
    # The binary search maximises font size, which minimises line count and
    # leaves large empty space top+bottom while text stretches left→right.
    # We step down in font size until the fill ratio reaches _MIN_FILL_RATIO
    # OR we have tried MAX_STEPS sizes OR we hit min_font_size.
    #
    # Edge cases handled:
    #   • Single-word text: skip (can't add lines, no improvement possible)
    #   • Explicit \n: user chose layout, skip
    #   • Already balanced: skip (ratio already ≥ threshold)
    MAX_STEPS = 12
    if (
        "\n" not in clean_text
        and clean_text.count(" ") >= 1
        and best_fit_lines_data
        and len(best_fit_lines_data) > 1
    ):
        n_lines = len(best_fit_lines_data)
        fill_ratio = _compute_fill_ratio(
            n_lines,
            best_fit_line_height,
            best_fit_max_line_width,
            max_render_width,
            max_render_height,
        )

        if fill_ratio < _MIN_FILL_RATIO and best_fit_size > min_font_size:
            lower_bound = max(min_font_size, best_fit_size - MAX_STEPS)
            log_message(
                f"Post-opt fill-ratio={fill_ratio:.2f} < {_MIN_FILL_RATIO}; "
                f"trying sizes {best_fit_size - 1} → {lower_bound}",
                verbose=verbose,
            )
            best_ratio_so_far = fill_ratio
            for try_size in range(best_fit_size - 1, lower_bound - 1, -1):
                fit_data = check_fit(
                    try_size,
                    clean_text,
                    max_render_width,
                    max_render_height,
                    regular_hb_face,
                    regular_typeface,
                    loaded_hb_faces,
                    features_to_enable,
                    line_spacing_mult,
                    hyphenate_before_scaling,
                    hyphen_penalty,
                    hyphenation_min_word_length,
                    badness_exponent,
                    word_width_cache,
                    verbose,
                    detach_trailing_ellipsis,
                )
                if fit_data is None:
                    continue
                new_ratio = _compute_fill_ratio(
                    len(fit_data["lines"]),
                    fit_data["line_height"],
                    fit_data["max_line_width"],
                    max_render_width,
                    max_render_height,
                )
                # Always accept if ratio improved — keep best seen so far
                if new_ratio > best_ratio_so_far:
                    best_ratio_so_far = new_ratio
                    best_fit_size = try_size
                    best_fit_lines_data = fit_data["lines"]
                    best_fit_metrics = fit_data["metrics"]
                    best_fit_max_line_width = fit_data["max_line_width"]
                    best_fit_line_height = fit_data["line_height"]
                    log_message(
                        f"Post-opt: size {try_size} ratio={new_ratio:.2f} "
                        f"({len(fit_data['lines'])} lines)",
                        verbose=verbose,
                    )
                # Stop as soon as ratio is balanced — no need to go smaller
                if new_ratio >= _MIN_FILL_RATIO:
                    log_message(
                        f"Post-opt: balanced at size {try_size} ratio={new_ratio:.2f}",
                        verbose=verbose,
                    )
                    break

    # ── Post-optimisation: final font-size shrink for breathing room ──────────
    # After the fill-ratio step-down, apply a uniform shrink so text has
    # comfortable margins from bubble edges.  Try reducing font size by ~8 %;
    # if the text still fits, keep the smaller size for a more balanced look.
    _FINAL_SHRINK = 0.88
    if best_fit_size > min_font_size:
        shrunk = max(min_font_size, int(best_fit_size * _FINAL_SHRINK))
        if shrunk < best_fit_size:
            fit_data = check_fit(
                shrunk, clean_text, max_render_width, max_render_height,
                regular_hb_face, regular_typeface, loaded_hb_faces,
                features_to_enable, line_spacing_mult,
                hyphenate_before_scaling, hyphen_penalty,
                hyphenation_min_word_length, badness_exponent,
                word_width_cache, verbose, detach_trailing_ellipsis,
            )
            if fit_data is not None:
                best_fit_size = shrunk
                best_fit_lines_data = fit_data["lines"]
                best_fit_metrics = fit_data["metrics"]
                best_fit_max_line_width = fit_data["max_line_width"]
                best_fit_line_height = fit_data["line_height"]
                log_message(
                    f"Final shrink: size {shrunk} fits",
                    verbose=verbose,
                )

    if best_fit_size < max_font_size:
        bubble_desc = f"bubble {bubble_id}" if bubble_id else "bubble"
        log_message(
            f"Shrinking text in {bubble_desc} to size {best_fit_size}",
            verbose=verbose,
        )

    # ── Compute exact block height from Skia metrics ──────────────────────────
    # block_height = actual pixel height of the full text block as Skia renders it.
    # Formula: cap_height for the first line + (n-1) * line_height for subsequent lines.
    # This is more accurate than n * line_height (which overcounts by one fLeading).
    # Used by psd_builder for precise vertical centering.
    best_fit_block_height = 0.0
    best_fit_ascent = 0.0
    if best_fit_metrics is not None and best_fit_lines_data:
        best_fit_ascent = -best_fit_metrics.fAscent
        cap_height = best_fit_ascent + best_fit_metrics.fDescent
        n = len(best_fit_lines_data)
        best_fit_block_height = cap_height + max(0, n - 1) * best_fit_line_height

    return {
        "font_size": best_fit_size,
        "lines": best_fit_lines_data,
        "metrics": best_fit_metrics,
        "max_line_width": best_fit_max_line_width,
        "line_height": best_fit_line_height,
        "block_height": best_fit_block_height,
        # ascent = pixels from baseline to top of glyphs (positive).
        # position_y in PhotoshopAPI = first-line baseline in pixels.
        # box_top = position_y - ascent  ->  text_center = box_top + content_h/2
        "ascent": best_fit_ascent,
    }
