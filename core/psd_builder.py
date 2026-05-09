"""
PSD export support for MangaTranslator.

Builds layered Photoshop PSD files with editable text layers per bubble.
Uses PhotoshopAPI for fully editable TypeLayers that render correctly in Photoshop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

try:
    import photoshopapi as psapi
    PSAPI_AVAILABLE = True
except ImportError:
    psapi = None  # type: ignore
    PSAPI_AVAILABLE = False

from utils.logging import log_message


# ---------------------------------------------------------------------------
# Helper: extract PostScript font name from a TTF/OTF file
# ---------------------------------------------------------------------------

def get_font_postscript_name(font_path: str) -> str:
    """
    Return the PostScript name embedded in a TrueType/OpenType font file.
    Falls back to the filename stem when the name table lacks a PS entry.
    """
    try:
        tt = TTFont(font_path)
        name_table = tt.get("name")
        if name_table is None:
            return Path(font_path).stem
        for rec in name_table.names:
            if rec.nameID == 6:  # PostScript name
                ps_name = rec.toUnicode()
                if ps_name:
                    return ps_name
        # Fallback: nameID 1 (font family)
        for rec in name_table.names:
            if rec.nameID == 1:
                family = rec.toUnicode()
                if family:
                    return family.replace(" ", "")
        return Path(font_path).stem
    except Exception:
        return Path(font_path).stem


# ---------------------------------------------------------------------------
# Helper: extract text pixels from before/after crop comparison
# ---------------------------------------------------------------------------

def _extract_text_delta(
    before_image: Image.Image,
    after_image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Extract changed pixels between before and after images in *bbox* region.

    Returns an RGBA image of the *bbox* size with text pixels opaque and
    unchanged pixels fully transparent.
    """
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw == 0 or bh == 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    before_crop = before_image.crop((x1, y1, x2, y2)).convert("RGBA")
    after_crop = after_image.crop((x1, y1, x2, y2)).convert("RGBA")

    before_arr = np.array(before_crop, dtype=np.int16)
    after_arr = np.array(after_crop, dtype=np.int16)

    changed = np.any(before_arr != after_arr, axis=-1)

    # Build RGBA output: use the after-image colour, alpha=255 where changed
    out = np.zeros((bh, bw, 4), dtype=np.uint8)
    out[changed] = np.clip(after_arr[changed], 0, 255).astype(np.uint8)
    out[changed, 3] = 255

    return Image.fromarray(out, mode="RGBA")


# ---------------------------------------------------------------------------
# Helpers for PhotoshopAPI
# ---------------------------------------------------------------------------

def _pil_to_psapi_chw(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGBA image to PhotoshopAPI's CHW uint8 format."""
    arr = np.array(pil_img.convert("RGBA"))
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def _to_argb_floats(rgb: Tuple[int, int, int]) -> List[float]:
    """Convert (R,G,B) 0-255 to [A,R,G,B] 0.0-1.0 floats (PhotoshopAPI format)."""
    return [1.0, rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]


def _set_text_antialias(layer: psapi.TextLayer_8bit) -> None:
    """Set non-soft anti-aliasing so text doesn't look washed out in PSD."""
    aa_enum = getattr(psapi.enum, "AntiAliasMethod", None)
    if aa_enum is None:
        return
    for name in ("Sharp", "Strong", "Crisp", "Smooth"):
        value = getattr(aa_enum, name, None)
        if value is not None:
            try:
                layer.set_anti_alias(value)
            except Exception:
                pass
            return


def _apply_text_style(
    layer: psapi.TextLayer_8bit,
    font_name: str,
    font_size: float,
    text_color: Tuple[int, int, int],
    bubble_class: str = "",
    line_height: Optional[float] = None,
    has_bold: bool = False,
    has_italic: bool = False,
) -> None:
    """Apply font, size, color, leading, and bold/italic to the default text style."""
    try:
        editor = layer.style_all()
    except Exception:
        return
    if editor is None:
        return
    try:
        editor.set_font(font_name)
        editor.set_font_size(font_size)
        editor.set_fill_color(_to_argb_floats(text_color))
        editor.set_auto_leading(False)
        # Ensure minimum 1.3× leading — Arabic fonts often have fLeading=0
        # which makes Skia line_height ≈ font_size, causing lines to collide.
        editor.set_leading(max(line_height, font_size * 1.3) if line_height else font_size * 1.3)
        if bubble_class:
            editor.set_bold(has_bold or bubble_class in ("box", "scream"))
            editor.set_italic(has_italic or bubble_class in ("scream", "thinking"))
    except Exception:
        pass


def _apply_paragraph_justification(layer: psapi.TextLayer_8bit) -> None:
    """Set center justification (standard for manga speech bubbles)."""
    just_enum = getattr(psapi.enum, "Justification", None)
    if just_enum is None:
        return
    center = getattr(just_enum, "Center", None)
    if center is None:
        return
    try:
        editor = layer.paragraph_all()
        if editor is not None:
            editor.set_justification(center)
    except Exception:
        pass


def _apply_style_ranges(
    layer: "psapi.TextLayer_8bit",
    style_segments: List[Tuple[int, int, str]],
    font_variant_paths: Dict[str, str],
    font_size: float,
    text_color: Tuple[int, int, int],
    line_height: Optional[float],
) -> None:
    """Apply per-character-range bold/italic styling using actual font variants.

    Parameters
    ----------
    style_segments      : list of (utf16_start, utf16_end, style_name) from psd_style_segments.
    font_variant_paths  : dict of style_name → font file path string.
    """
    if not style_segments:
        return

    _STYLE_FLAGS = {
        "bold":       (True,  False),
        "italic":     (False, True),
        "bold_italic": (True,  True),
    }
    _ps_name_cache: Dict[str, str] = {}

    for start, end, style in style_segments:
        if start >= end:
            continue
        try:
            rng = layer.style_range(start, end)
            if rng is None or not rng.valid:
                continue

            # Prefer actual font variant file; fall back to faux bold/italic.
            variant_path = font_variant_paths.get(style)
            if variant_path:
                if variant_path not in _ps_name_cache:
                    _ps_name_cache[variant_path] = get_font_postscript_name(variant_path)
                ps_name = _ps_name_cache[variant_path]
                rng.set_font(ps_name)
            else:
                is_bold, is_italic = _STYLE_FLAGS.get(style, (False, False))
                rng.set_bold(is_bold)
                rng.set_italic(is_italic)

            rng.set_font_size(font_size)
            rng.set_fill_color(_to_argb_floats(text_color))
            rng.set_auto_leading(False)
            rng.set_leading(max(line_height, font_size * 1.3) if line_height else font_size * 1.3)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Photoshop post-processing (run JSX script on saved PSD)
# ---------------------------------------------------------------------------

def run_photoshop_postprocess(psd_path: Path, jsx_script_path: str = "") -> None:
    """Open the saved PSD in Photoshop and run a JSX post-processing script on it.

    Requires Photoshop to be installed and the ``pywin32`` package.
    Silently skipped when *jsx_script_path* is empty, the file is missing,
    pywin32 is not installed, or Photoshop COM is unavailable.
    """
    if not jsx_script_path:
        return
    jsx_path = Path(jsx_script_path)
    if not jsx_path.exists():
        log_message(f"  Warning: JSX script not found: {jsx_path}", always_print=True)
        return
    try:
        import win32com.client
    except ImportError:
        log_message(
            "  Warning: pywin32 not installed. Cannot run Photoshop post-processing.",
            always_print=True,
        )
        return

    try:
        ps = win32com.client.Dispatch("Photoshop.Application")
        log_message(
            "  Opening PSD in Photoshop for post-processing...",
            always_print=True,
        )
        doc = ps.Open(str(psd_path))

        with open(jsx_path, "r", encoding="utf-8-sig") as f:
            script = f.read()

        log_message(f"  Running JSX script: {jsx_path.name}", always_print=True)
        ps.DoJavaScript(script)

        doc.Save()
        doc.Close()
        log_message("  Photoshop post-processing complete.", always_print=True)
    except Exception as e:
        log_message(
            f"  Warning: Photoshop post-processing failed: {e}",
            always_print=True,
        )


# ---------------------------------------------------------------------------
# Main PSD builder
# ---------------------------------------------------------------------------

def build_psd(
    original_image: Image.Image,
    clean_background: Image.Image,
    text_layers_info: List[Dict[str, Any]],
    output_path: Path,
    target_size: Optional[Tuple[int, int]] = None,
    verbose: bool = False,
) -> None:
    """
    Assemble and save a layered Photoshop PSD with editable text layers.

    Uses PhotoshopAPI for fully editable TypeLayers.

    Layer stack (top to bottom):
      "Editable Text" group -> individual Bubble text layers
      "Cleaned Background"
      "Original" (optional)

    Parameters
    ----------
    original_image    : The source page as loaded from disk (RGB/RGBA).
    clean_background  : The inpainted page with original text removed.
    text_layers_info  : List of per-bubble dicts:
                        - text       : str
                        - bbox       : (x1, y1, x2, y2)
                        - font_name  : str (PostScript name)
                        - font_size  : float
                        - text_color : (R,G,B) or None
                        - text_image : PIL.Image (optional, unused in psapi path)
    output_path       : Where to write the .psd file.
    target_size       : Optional (width, height) to resize layers to.
    verbose           : Emit progress messages.
    """
    output_path = Path(output_path)

    if not PSAPI_AVAILABLE:
        raise ImportError(
            "photoshopapi is required for PSD export. "
            "Install with: pip install photoshopapi"
        )

    # ── 1. Normalise layers ─────────────────────────────────────────────
    log_message("Building PSD layers\u2026", verbose=verbose, always_print=True)

    clean_rgba = clean_background.convert("RGBA")
    ref_size = clean_rgba.size
    w, h = ref_size

    # Optional target resize
    if target_size is not None and target_size != ref_size:
        log_message(
            f"Resizing PSD layers {ref_size} \u2192 {target_size}",
            verbose=verbose,
            always_print=True,
        )
        clean_rgba = clean_rgba.resize(target_size, Image.Resampling.LANCZOS)
        w, h = target_size

    # ── 2. Create document ──────────────────────────────────────────────
    doc = psapi.LayeredFile_8bit(psapi.enum.ColorMode.rgb, w, h)
    doc.dpi = 300.0

    # PhotoshopAPI add_layer(): first-added is topmost.
    # Build order so the final stack is:
    #   Top:    Editable Text group
    #   Middle: Cleaned Background
    #   Bottom: Original (optional)

    # --- Top: Editable Text group (added first = topmost) ---
    text_group = psapi.GroupLayer_8bit("Editable Text")
    doc.add_layer(text_group)

    for i, info in enumerate(text_layers_info):
        text = info.get("text", "")
        if not text:
            continue

        bbox = info.get("bbox", (0, 0, 0, 0))
        font_name = info.get("font_name", "ArialMT")
        font_size = info.get("font_size", 16.0)
        text_color = info.get("text_color", (0, 0, 0))
        # Handle case where key exists but value is None
        if text_color is None:
            text_color = (0, 0, 0)
        else:
            # Snap text color to pure black/white — no grey in PSD
            tb = (text_color[0] + text_color[1] + text_color[2]) / 3.0
            text_color = (0, 0, 0) if tb < 128 else (255, 255, 255)
        x1, y1, x2, y2 = bbox
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)

        # ── Positioning: EXACT same math as drawing_engine.py (PNG path) ─────
        #
        # drawing_engine centers text with:
        #   total_visual_height = (n-1)*line_height - fAscent + fDescent
        #   block_top_y         = cy - total_visual_height / 2
        #   (+6% font_size shift for Arabic diacritics)
        #   first_baseline_y    = block_top_y - fAscent
        #
        # For PhotoshopAPI TextLayer:
        #   position_x = LEFT  edge of text box (pixels)
        #   position_y = FIRST LINE BASELINE     (pixels)
        # → position_y = first_baseline_y  (same value, directly usable)
        #
        # IMPORTANT: No fill_scale for PSD. Skia metrics ≠ Photoshop metrics,
        # so scaling amplifies positioning errors. Use Skia's natural size;
        # the user can adjust in Photoshop.
        psd_lines = info.get("psd_text_lines")
        if psd_lines:
            psd_text  = psd_lines
            cx        = info.get("psd_text_center_x")
            cy        = info.get("psd_text_center_y")
            max_w     = info.get("psd_text_max_width", 0.0)
            line_h    = info.get("psd_text_line_height", 0.0)
            f_ascent  = info.get("psd_text_fAscent",  0.0)   # negative (Skia)
            f_descent = info.get("psd_text_fDescent", 0.0)   # positive (Skia)
            num_lines = psd_text.count("\r") + 1

            content_w = float(max_w) if max_w else float(bw)
            # total_visual_height mirrors drawing_engine exactly
            if line_h and (f_ascent or f_descent):
                total_visual_height = (
                    (num_lines - 1) * float(line_h)
                    - float(f_ascent)          # fAscent negative → subtracted = +
                    + float(f_descent)
                )
            else:
                total_visual_height = float(info.get("psd_text_block_height", float(bh) * 0.5))

            bubble_cx = float(cx) if cx is not None else float(x1 + bw / 2)
            bubble_cy = float(cy) if cy is not None else float(y1 + bh / 2)

            # block_top_y (same as drawing_engine)
            block_top_y = bubble_cy - total_visual_height / 2.0

            # Arabic diacritics shift
            # PSD: 15% of ascent (not font_size).
            # Photoshop's ascent for Arabic fonts is typically larger than Skia's,
            # so the text appears too high. Using ascent-based shift adapts to
            # each font's actual diacritic space.
            # Non-Arabic text gets NO shift — same as drawing_engine.py.
            if any(0x0600 <= ord(c) <= 0x06FF for c in psd_text):
                block_top_y += -float(f_ascent) * 0.15

            # first_baseline_y = block_top_y - fAscent
            # fAscent is negative so subtracting it moves DOWN (correct)
            first_baseline_y = block_top_y - float(f_ascent)

            # box: content-sized with small padding
            # width uses max of content or 60% bubble (room for minor edits)
            # height is content height + padding proportional to font size
            box_padding_h = float(font_size) * 0.5
            box_padding_w = float(font_size) * 0.5
            box_w = max(content_w + 2.0 * box_padding_w, float(bw) * 0.6)
            box_h = total_visual_height + 2.0 * box_padding_h

            # position_x = LEFT edge so center-aligned text is centred at cx
            pos_x = bubble_cx - box_w / 2.0
            # position_y = first line baseline (proven by PSD binary analysis)
            pos_y = first_baseline_y

        else:
            psd_text = text
            line_h   = None
            box_w    = float(bw)
            box_h    = float(bh)
            pos_x    = float(x1)
            pos_y    = float(y1 + bh / 2)


        log_message(
            f"  Adding text layer '{text[:24]}\u2026' for bubble {bbox}",
            verbose=verbose,
        )

        try:
            layer = psapi.TextLayer_8bit(
                layer_name=f"Bubble {i + 1}",
                text=psd_text,
                font=font_name,
                font_size=float(font_size),
                fill_color=_to_argb_floats(text_color),
                position_x=pos_x,
                position_y=pos_y,
                box_width=box_w,
                box_height=box_h,
            )
            _set_text_antialias(layer)
            _apply_text_style(
                layer, font_name, font_size, text_color,
                bubble_class=info.get("psd_bubble_class", ""),
                # line_h is set in the psd_lines branch;
                # fall back to original stored value in the else branch.
                line_height=line_h if psd_lines else info.get("psd_text_line_height"),
                has_bold=info.get("psd_text_has_bold", False),
                has_italic=info.get("psd_text_has_italic", False),
            )
            _apply_paragraph_justification(layer)

            # Apply per-character-range bold/italic using real font variants.
            _apply_style_ranges(
                layer,
                style_segments=info.get("psd_style_segments", []),
                font_variant_paths=info.get("psd_font_variant_paths", {}),
                font_size=float(font_size),
                text_color=text_color,
                line_height=info.get("psd_text_line_height"),
            )

            text_group.add_layer(doc, layer)
            layer.fill = 1.0
            layer.opacity = 1.0
        except Exception as exc:
            log_message(
                f"  Warning: failed to create text layer for bubble {i}: {exc}",
                verbose=verbose,
                always_print=True,
            )

    # --- Middle: Cleaned Background ---
    log_message("  Adding 'Cleaned Background' layer...", verbose=verbose)
    clean_layer = psapi.ImageLayer_8bit(
        _pil_to_psapi_chw(clean_rgba),
        "Cleaned Background",
        width=w,
        height=h,
        pos_x=w / 2.0,
        pos_y=h / 2.0,
    )
    clean_layer.fill = 1.0
    clean_layer.opacity = 1.0
    doc.add_layer(clean_layer)

    # --- Bottom: Original (optional) ---
    if original_image is not None:
        orig = original_image.convert("RGBA")
        if orig.size != (w, h):
            orig = orig.resize((w, h), Image.Resampling.LANCZOS)
        log_message("  Adding 'Original' layer...", verbose=verbose)
        orig_layer = psapi.ImageLayer_8bit(
            _pil_to_psapi_chw(orig),
            "Original",
            width=w,
            height=h,
            pos_x=w / 2.0,
            pos_y=h / 2.0,
        )
        orig_layer.fill = 1.0
        orig_layer.opacity = 1.0
        doc.add_layer(orig_layer)

    # Force Photoshop to re-render text layers on open
    invalidate = getattr(doc, "invalidate_text_cache", None)
    if callable(invalidate):
        invalidate()

    # ── Save ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_message(f"Saving PSD \u2192 {output_path}", verbose=verbose, always_print=True)
    doc.write(str(output_path), force_overwrite=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log_message(
        f"PSD saved ({size_mb:.1f} MB): {output_path}",
        verbose=verbose,
        always_print=True,
    )
