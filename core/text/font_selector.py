"""
font_selector.py — Bubble-type aware font directory resolution.

Maps YOLO detection class names to font directories using a two-step strategy:
  1. Exact key match  (user-defined mapping in config)
  2. Keyword fallback (built-in heuristics for common YOLO class names)

Bubble categories (from user's style guide):
  • Normal / dialogue  → default font_dir (already set in RenderingConfig)
  • Thought / cloud    → font_dir_thought
  • Scream / jagged    → font_dir_scream
  • Square / narrative → font_dir_square
  • Outside bubble     → already handled by osb_font_dir
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# ── Keyword sets for fuzzy matching ──────────────────────────────────────────
# Each canonical type maps to a list of substrings that might appear in a YOLO
# class name (case-insensitive).  Add more synonyms here as needed.

_THOUGHT_KEYWORDS = [
    "thought", "think", "cloud", "dream", "bubble_thought",
    # Arabic transliterations that might appear in custom models
    "tafkeer", "tafkir",
]

_SCREAM_KEYWORDS = [
    "scream", "shout", "yell", "loud", "burst", "spike", "angry",
    "exclaim", "spiky", "jagged", "action",
    "sarakh", "sarkha",
]

_SQUARE_KEYWORDS = [
    "square", "box", "rect", "narrat", "caption", "system",
    "sign", "label", "note", "description",
    "morabbe", "mrabba",
]

# Internal canonical keys
_TYPE_THOUGHT = "thought"
_TYPE_SCREAM  = "scream"
_TYPE_SQUARE  = "square"
_TYPE_NORMAL  = "normal"   # fallback / default


def _classify_bubble(class_name: str) -> str:
    """
    Return one of the four canonical type keys based on the YOLO class name.
    Uses keyword matching; returns 'normal' if nothing matches.
    """
    cn = class_name.lower().strip()

    # Remove separators so "bubble_thought" → "bubblethought"
    cn_flat = re.sub(r"[_\-\s]+", "", cn)

    for kw in _THOUGHT_KEYWORDS:
        if kw in cn or kw in cn_flat:
            return _TYPE_THOUGHT

    for kw in _SCREAM_KEYWORDS:
        if kw in cn or kw in cn_flat:
            return _TYPE_SCREAM

    for kw in _SQUARE_KEYWORDS:
        if kw in cn or kw in cn_flat:
            return _TYPE_SQUARE

    return _TYPE_NORMAL


def resolve_font_dir(
    bubble_class: str,
    font_dir_map: Dict[str, str],
    default_font_dir: str,
) -> str:
    """
    Select the appropriate font directory for a given bubble class.

    Resolution order
    ----------------
    1. Exact match:      font_dir_map.get(bubble_class)
    2. Canonical match:  font_dir_map.get(_classify_bubble(bubble_class))
    3. Default:          default_font_dir

    Parameters
    ----------
    bubble_class    : YOLO class string, e.g. "thought", "bubble_scream", "text"
    font_dir_map    : mapping of canonical keys / class names → font directory paths.
                      Recognised canonical keys: "thought", "scream", "square".
                      Example::

                          {
                              "thought": "fonts/Al_Hor",
                              "scream":  "fonts/Kharabeesh",
                              "square":  "fonts/Hacen_Tunisia",
                          }

    default_font_dir: fallback path (the normal-dialogue font directory)

    Returns
    -------
    str — resolved font directory path
    """
    if not bubble_class or not font_dir_map:
        return default_font_dir

    # 1. Exact key match
    if bubble_class in font_dir_map:
        return font_dir_map[bubble_class]

    # 2. Canonical type match (keyword-based)
    canonical = _classify_bubble(bubble_class)
    if canonical != _TYPE_NORMAL and canonical in font_dir_map:
        return font_dir_map[canonical]

    return default_font_dir


def describe_mapping(font_dir_map: Dict[str, str], default_font_dir: str) -> str:
    """Return a human-readable summary of the active font mapping (for logging)."""
    lines = [f"  normal/dialogue → {default_font_dir}"]
    for key in [_TYPE_THOUGHT, _TYPE_SCREAM, _TYPE_SQUARE]:
        if key in font_dir_map:
            lines.append(f"  {key:8s}        → {font_dir_map[key]}")
    return "\n".join(lines)
