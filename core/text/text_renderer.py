from typing import Optional, Tuple

import numpy as np
import skia
from PIL import Image

from core.config import RenderingConfig
from core.image.image_utils import calculate_centroid_expansion_box
from core.text.drawing_engine import (
    draw_layout,
    load_font_resources,
    pil_to_skia_surface,
    skia_surface_to_pil,
)
from core.text.font_manager import (
    find_font_variants,
    get_font_features,
    sanitize_text_for_font,
)
from core.text.font_selector import describe_mapping, resolve_font_dir
from core.text.layout_engine import find_optimal_layout
from core.text.text_processing import STYLE_PATTERN, convert_linebreak_marker, parse_styled_segments
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

GRAYSCALE_MIDPOINT = 128
FALLBACK_PADDING_RATIO = 0.15


def _store_psd_layout_info(
    image: Image.Image,
    layout_data: dict,
    center_x: float,
    center_y: float,
    font_variants: Optional[dict] = None,
) -> None:
    lines = layout_data.get("lines", [])
    line_texts = []
    has_bold = False
    has_italic = False
    # Track per-character style segments for PSD range styling.
    # Each entry: (utf16_start, utf16_end, style_name)
    # style_name one of: "italic", "bold", "bold_italic"
    style_segments = []
    char_offset = 0

    for i, line in enumerate(lines):
        text_with_markers = line.get("text_with_markers", "")
        for match in STYLE_PATTERN.finditer(text_with_markers):
            marker = match.group(1)
            if marker == "***":
                has_bold = True
                has_italic = True
            elif marker == "**":
                has_bold = True
            elif marker == "*":
                has_italic = True
        plain = STYLE_PATTERN.sub(r"\2", text_with_markers)
        line_texts.append(plain)

        # Build per-segment records for this line
        segs = parse_styled_segments(text_with_markers)
        for seg_text, seg_style in segs:
            # Use UTF-16 code-unit length (PhotoshopAPI requirement).
            # Astral-plane characters count as 2; BMP as 1.
            seg_utf16_len = sum(2 if ord(c) > 0xFFFF else 1 for c in seg_text)
            if seg_style != "regular":
                style_segments.append(
                    (char_offset, char_offset + seg_utf16_len, seg_style)
                )
            char_offset += seg_utf16_len

        if i < len(lines) - 1:
            char_offset += 1  # newline is 1 UTF-16 code unit

    image.info["psd_text_lines"] = "\r".join(line_texts)
    image.info["psd_text_max_width"] = layout_data.get("max_line_width", 0.0)
    image.info["psd_text_line_height"] = layout_data.get("line_height", 0.0)
    # Exact Skia-measured block height: cap_height + (n-1)*line_height.
    # More accurate than n*line_height — used by psd_builder for centering.
    image.info["psd_text_block_height"] = layout_data.get("block_height", 0.0)
    # Store raw Skia metrics so psd_builder can mirror drawing_engine math exactly
    metrics = layout_data.get("metrics")
    image.info["psd_text_fAscent"]  = float(metrics.fAscent)  if metrics else 0.0
    image.info["psd_text_fDescent"] = float(metrics.fDescent) if metrics else 0.0
    image.info["psd_text_ascent"] = layout_data.get("ascent", 0.0)
    image.info["psd_text_center_x"] = center_x
    image.info["psd_text_center_y"] = center_y
    image.info["psd_text_has_bold"] = has_bold
    image.info["psd_text_has_italic"] = has_italic
    image.info["psd_style_segments"] = style_segments  # NEW: per-range style info

    # Store font variant file paths so psd_builder can look up PostScript names.
    if font_variants:
        image.info["psd_font_variant_paths"] = {
            style: str(path)
            for style, path in font_variants.items()
            if path
        }
    else:
        image.info["psd_font_variant_paths"] = {}


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    config: Optional[RenderingConfig] = None,
    raise_on_safe_error: bool = False,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    rotation_deg: float = 0.0,
    vertical_stack: bool = False,
    text_color_rgb: Optional[Tuple[int, int, int]] = None,
    text_background_color: Optional[Tuple[int, int, int]] = None,
    layout_only: bool = False,
    is_rtl: bool = False,
    bubble_class: str = "",
) -> Image.Image:
    if config is None:
        config = RenderingConfig()

    if bubble_class and getattr(config, "font_dir_map", None):
        resolved = resolve_font_dir(bubble_class, config.font_dir_map, font_dir)
        if resolved != font_dir:
            log_message(
                f"Font override for bubble class '{bubble_class}': {resolved}",
                verbose=verbose,
            )
            font_dir = resolved
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid bbox dimensions: {bbox}", always_print=True)
        raise RenderingError(f"Invalid bounding box dimensions: {bbox}")

    normalized_text = text.replace("\u2014", "-")

    # Convert \\n marker to actual newlines for forced line breaks
    normalized_text = convert_linebreak_marker(normalized_text)

    # Preserve explicit newlines while normalizing whitespace within each segment
    segments = normalized_text.split("\n")
    clean_segments = []
    for seg in segments:
        cleaned = " ".join(seg.split())
        clean_segments.append(cleaned)
    clean_text = "\n".join(clean_segments)
    if not clean_text.strip():
        return pil_image

    if vertical_stack:
        import unicodedata

        def _is_separator_or_space(ch: str) -> bool:
            try:
                cat = unicodedata.category(ch)
            except Exception:
                return ch.isspace()
            return ch.isspace() or (len(cat) > 0 and cat[0] == "Z")

        stacked_chars = [ch for ch in clean_text if not _is_separator_or_space(ch)]
        layout_text = "\n".join(stacked_chars)
    else:
        layout_text = clean_text

    layout_box_top_left = None
    safe_area_result = None
    safe_area_fallback_logged = False
    if cleaned_mask is not None:
        try:
            # Proportional padding: scale with bubble size so small bubbles get
            # tighter margins and large bubbles get more breathing room.
            # Use at least the configured minimum padding.
            prop_padding = max(bubble_width, bubble_height) * 0.15
            effective_padding = max(prop_padding, config.padding_pixels)
            safe_area_result = calculate_centroid_expansion_box(
                cleaned_mask, padding_pixels=effective_padding, verbose=verbose,
                target_center=(x1 + bubble_width / 2.0, y1 + bubble_height / 2.0),
            )
        except ImageProcessingError:
            safe_area_result = None
            if raise_on_safe_error:
                raise
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
            safe_area_fallback_logged = True

    if safe_area_result is not None:
        guaranteed_box, _ = safe_area_result
        box_x, box_y, box_w, box_h = guaranteed_box
        # Enforce square safe area: largest inscribed square centered in the box.
        # This prevents text from spilling outside the bubble in oval/irregular shapes.
        box_size = min(box_w, box_h)
        offset_x = (box_w - box_size) // 2
        offset_y = (box_h - box_size) // 2
        box_x += offset_x
        box_y += offset_y
        box_w = box_size
        box_h = box_size
        layout_box_top_left = (box_x, box_y)
        max_render_width = float(box_w)
        max_render_height = float(box_h)
        target_center_x = box_x + box_w / 2.0
        target_center_y = box_y + box_h / 2.0
        log_message("Using centroid-based safe area calculation (squared)", verbose=verbose)
    else:
        if not safe_area_fallback_logged:
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
        max_render_width = bubble_width * (1 - 2 * FALLBACK_PADDING_RATIO)
        max_render_height = bubble_height * (1 - 2 * FALLBACK_PADDING_RATIO)

        if max_render_width <= 0 or max_render_height <= 0:
            max_render_width = max(1.0, float(bubble_width))
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0

    try:
        font_variants = find_font_variants(font_dir, verbose=verbose)
        regular_font_path = font_variants.get("regular")
    except FontError as e:
        raise RenderingError(f"Font loading failed: {e}") from e

    _BUBBLE_TYPE_STYLE = {"box": "bold", "scream": "bold_italic", "thinking": "italic"}
    _forced_style = _BUBBLE_TYPE_STYLE.get(bubble_class)
    if _forced_style:
        _target_path = font_variants.get(_forced_style) or font_variants.get("regular")
        if _target_path and _target_path != font_variants.get("regular"):
            font_variants = dict(font_variants)
            for _k in ("regular", "italic", "bold", "bold_italic"):
                font_variants[_k] = _target_path
            regular_font_path = _target_path
            log_message(
                f"Bubble type '{bubble_class}' \u2192 using {_forced_style} font: {_target_path.name}",
                verbose=verbose,
            )

    layout_text = sanitize_text_for_font(
        layout_text, str(regular_font_path), verbose=verbose
    )
    if not layout_text.strip():
        log_message(
            "All text characters unsupported by font, skipping render",
            always_print=True,
        )
        return pil_image

    try:
        _, regular_typeface, regular_hb_face = load_font_resources(
            str(regular_font_path)
        )
    except FontError as e:
        raise RenderingError(f"Font resource loading failed: {e}") from e

    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        "kern": "kern" in available_features["GPOS"],
        "liga": config.use_ligatures and "liga" in available_features["GSUB"],
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(
        f"Font features: {[k for k, v in features_to_enable.items() if v]}",
        verbose=verbose,
    )

    preload_hb_faces = {"regular": regular_hb_face}
    for style_key in ["italic", "bold", "bold_italic"]:
        style_path = font_variants.get(style_key)
        if style_path:
            _, _typeface, _hb_face = load_font_resources(str(style_path))
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    try:
        layout_data = find_optimal_layout(
            layout_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            config.min_font_size,
            config.max_font_size,
            config.line_spacing_mult,
            False if vertical_stack else config.hyphenate_before_scaling,
            config.hyphen_penalty,
            config.hyphenation_min_word_length,
            config.badness_exponent,
            verbose,
            bubble_id,
            cleaned_mask,
            layout_box_top_left,
            config.detach_trailing_ellipsis,
        )
    except RenderingError as e:
        raise RenderingError(f"Layout optimization failed: {e}") from e

    if layout_only:
        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        result = Image.new("RGBA", (1, 1))
        result.info["font_size"] = layout_data["font_size"]
        return result

    required_styles = {"regular"} | {
        style for _, style in parse_styled_segments(clean_text)
    }
    log_message(f"Required styles: {sorted(required_styles)}", verbose=verbose)

    loaded_typefaces = {"regular": regular_typeface}
    loaded_hb_faces = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style}: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = load_font_resources(str(font_path))
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Failed to load {style} variant, using regular",
                        verbose=verbose,
                    )
            else:
                log_message(
                    f"Style '{style}' not found, using regular",
                    verbose=verbose,
                )

    text_color = skia.ColorBLACK
    if text_color_rgb is not None:
        text_color = skia.Color(text_color_rgb[0], text_color_rgb[1], text_color_rgb[2])
    else:
        sampled_brightness: Optional[float] = None
        try:
            img_w, img_h = pil_image.size
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(img_w, x2)
            cy2 = min(img_h, y2)

            if cx2 > cx1 and cy2 > cy1:
                crop_np = np.array(pil_image.crop((cx1, cy1, cx2, cy2)))

                if crop_np.ndim == 3 and crop_np.shape[2] == 4:
                    gray_crop = np.dot(crop_np[..., :3], [0.299, 0.587, 0.114])
                elif crop_np.ndim == 3:
                    gray_crop = np.dot(crop_np, [0.299, 0.587, 0.114])
                else:
                    gray_crop = crop_np.astype(float)

                if cleaned_mask is not None:
                    mask_h, mask_w = cleaned_mask.shape[:2]
                    mx1 = max(0, x1)
                    my1 = max(0, y1)
                    mx2 = min(mask_w, x2)
                    my2 = min(mask_h, y2)
                    mask_crop = cleaned_mask[my1:my2, mx1:mx2]

                    if mask_crop.shape != gray_crop.shape:
                        import cv2 as _cv2
                        mask_crop = _cv2.resize(
                            mask_crop,
                            (gray_crop.shape[1], gray_crop.shape[0]),
                            interpolation=_cv2.INTER_NEAREST,
                        )

                    inside = gray_crop[mask_crop > 127]
                    if inside.size > 0:
                        sampled_brightness = float(np.median(inside))
                else:
                    h, w = gray_crop.shape
                    ih = max(1, h // 4)
                    iw = max(1, w // 4)
                    inner = gray_crop[ih: h - ih, iw: w - iw]
                    if inner.size > 0:
                        sampled_brightness = float(np.median(inner))
        except Exception:
            sampled_brightness = None

        if sampled_brightness is not None:
            text_color = (
                skia.ColorWHITE
                if sampled_brightness < GRAYSCALE_MIDPOINT
                else skia.ColorBLACK
            )
        elif bubble_color_bgr is not None:
            try:
                bg_brightness = (
                    bubble_color_bgr[0] + bubble_color_bgr[1] + bubble_color_bgr[2]
                ) / 3.0
                text_color = (
                    skia.ColorWHITE
                    if bg_brightness < GRAYSCALE_MIDPOINT
                    else skia.ColorBLACK
                )
            except Exception:
                text_color = skia.ColorBLACK

    skia_bg_color = None
    if text_background_color is not None:
        skia_bg_color = skia.Color(
            text_background_color[0],
            text_background_color[1],
            text_background_color[2],
        )

    if config.supersampling_factor > 1:
        log_message(
            f"Using supersampling factor {config.supersampling_factor}", verbose=verbose
        )

        img_width, img_height = pil_image.size
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(img_width, x2)
        crop_y2 = min(img_height, y2)

        cropped_region = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        factor = config.supersampling_factor
        scaled_width = int(crop_width * factor)
        scaled_height = int(crop_height * factor)
        upscaled_region = cropped_region.resize(
            (scaled_width, scaled_height), Image.Resampling.LANCZOS
        )

        scaled_target_center_x = (target_center_x - crop_x1) * factor
        scaled_target_center_y = (target_center_y - crop_y1) * factor

        scaled_layout_data = layout_data.copy()
        scaled_layout_data["font_size"] = layout_data["font_size"] * factor
        scaled_layout_data["line_height"] = layout_data["line_height"] * factor
        scaled_layout_data["max_line_width"] = layout_data["max_line_width"] * factor

        for line_data in scaled_layout_data["lines"]:
            line_data["width"] = line_data["width"] * factor

        original_metrics = layout_data["metrics"]

        class ScaledMetrics:
            def __init__(self, original, scale_factor):
                self.fAscent = original.fAscent * scale_factor
                self.fDescent = original.fDescent * scale_factor
                if hasattr(original, "fLeading"):
                    self.fLeading = original.fLeading * scale_factor
                if hasattr(original, "fXMin"):
                    self.fXMin = original.fXMin * scale_factor
                if hasattr(original, "fXMax"):
                    self.fXMax = original.fXMax * scale_factor
                if hasattr(original, "fYMin"):
                    self.fYMin = original.fYMin * scale_factor
                if hasattr(original, "fYMax"):
                    self.fYMax = original.fYMax * scale_factor

        scaled_metrics = ScaledMetrics(original_metrics, factor)
        scaled_layout_data["metrics"] = scaled_metrics

        try:
            scaled_surface = pil_to_skia_surface(upscaled_region)
        except RenderingError as e:
            raise RenderingError(f"Scaled surface preparation failed: {e}") from e

        success = draw_layout(
            scaled_surface,
            scaled_layout_data,
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_x
            ),
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_y
            ),
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            config.outline_width * factor,
            verbose,
            pre_translate_x=(
                float(scaled_target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(scaled_target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        try:
            scaled_pil_result = skia_surface_to_pil(scaled_surface)
        except RenderingError as e:
            raise RenderingError(f"Scaled conversion failed: {e}") from e

        downscaled_result = scaled_pil_result.resize(
            (crop_width, crop_height), Image.Resampling.LANCZOS
        )

        final_pil_image = pil_image.copy()
        final_pil_image.paste(downscaled_result, (crop_x1, crop_y1))

        log_message(
            f"Rendered at size {layout_data['font_size']} with {factor}x supersampling",
            verbose=verbose,
        )
        final_pil_image.info["font_size"] = layout_data["font_size"]
        final_pil_image.info["psd_font_path"] = str(regular_font_path)
        _store_psd_layout_info(final_pil_image, layout_data, target_center_x, target_center_y, font_variants=font_variants)
        return final_pil_image
    else:
        try:
            surface = pil_to_skia_surface(pil_image)
        except RenderingError as e:
            raise RenderingError(f"Surface preparation failed: {e}") from e

        success = draw_layout(
            surface,
            layout_data,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_x,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_y,
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            config.outline_width,
            verbose,
            pre_translate_x=(
                float(target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        try:
            final_pil_image = skia_surface_to_pil(surface)
        except RenderingError as e:
            raise RenderingError(f"Final conversion failed: {e}") from e

        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        final_pil_image.info["font_size"] = layout_data["font_size"]
        final_pil_image.info["psd_font_path"] = str(regular_font_path)
        _store_psd_layout_info(final_pil_image, layout_data, target_center_x, target_center_y, font_variants=font_variants)
        return final_pil_image
