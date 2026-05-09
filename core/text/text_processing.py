import re
from typing import Callable, List, Optional, Tuple

import numpy as np

# Markdown-like style pattern: ***bold italic***, **bold**, *italic*
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")

# Linebreak marker: when the translation contains literal "\\n" (backslash-n),
# it gets converted to an actual newline for forced line breaks.
LINEBREAK_MARKER = "\\n"


def convert_linebreak_marker(text: str) -> str:
    """Convert ``\\n`` marker to actual ``\\n`` for forced line breaks."""
    return text.replace(LINEBREAK_MARKER, "\n")


def is_rtl_script(text: str) -> bool:
    """Check if text contains dominant RTL script characters (Arabic, Hebrew, etc.)."""
    rtl_count = 0
    ltr_count = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in ("*",):
            continue
        # Arabic (0600–06FF), Arabic Supplement (0750–077F), Arabic Extended-A (08A0–08FF),
        # Arabic Presentation Forms A/B (FB50–FDFF, FE70–FEFF)
        if (
            0x0600 <= cp <= 0x06FF
            or 0x0750 <= cp <= 0x077F
            or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF
            or 0xFE70 <= cp <= 0xFEFF
        ):
            rtl_count += 1
        # Hebrew (0590–05FF, FB1D–FB4F)
        elif 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
            rtl_count += 1
        # Thaana (0780–07BF) — Maldivian RTL
        elif 0x0780 <= cp <= 0x07BF:
            rtl_count += 1
        # NKo (07C0–07FA) — Mande RTL
        elif 0x07C0 <= cp <= 0x07FA:
            rtl_count += 1
        else:
            ltr_count += 1
    return rtl_count > ltr_count


def is_latin_style_language(language_name: str) -> bool:
    """
    Determines if a language typically uses Latin script and hyphenation.
    This is used to decide whether to apply automatic hyphenation logic.
    """
    latin_style_languages = {
        "english",
        "french",
        "spanish",
        "german",
        "italian",
        "portuguese",
        "dutch",
        "polish",
        "czech",
        "swedish",
        "danish",
        "norwegian",
        "finnish",
        "hungarian",
        "romanian",
        "turkish",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
    }
    return language_name.lower() in latin_style_languages


def _is_hangul_character(char: str) -> bool:
    """Check if a character is Korean Hangul (syllables or jamo)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
    )


def is_cjk_character(char: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0x4E00 <= code <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= code <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= code <= 0x2CEAF)  # CJK Extension B-F
        or (0xF900 <= code <= 0xFAFF)  # CJK Compatibility
        or (0x3040 <= code <= 0x309F)  # Hiragana
        or (0x30A0 <= code <= 0x30FF)  # Katakana
        or (0x31F0 <= code <= 0x31FF)  # Katakana Extensions
        or (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
        or (0x3000 <= code <= 0x303F)  # CJK Symbols/Punctuation
        or (0xFF00 <= code <= 0xFFEF)  # Fullwidth Forms
    )


def parse_styled_segments(text: str) -> List[Tuple[str, str]]:
    """
    Parses text with markdown-like style markers into segments.

    Args:
        text (str): Input text potentially containing ***bold italic***, **bold**, *italic*.

    Returns:
        List[Tuple[str, str]]: List of (segment_text, style_name) tuples.
                               style_name is one of "regular", "italic", "bold", "bold_italic".
    """
    segments = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        marker = match.group(1)
        content = match.group(2)

        if start > last_end:
            segments.append((text[last_end:start], "regular"))

        style = "regular"
        if len(marker) == 3:
            style = "bold_italic"
        elif len(marker) == 2:
            style = "bold"
        elif len(marker) == 1:
            style = "italic"

        segments.append((content, style))
        last_end = end

    if last_end < len(text):
        segments.append((text[last_end:], "regular"))

    return [(txt, style) for txt, style in segments if txt]


# Kinsoku Shori (禁則処理) - CJK line-breaking rules
KINSOKU_NOT_AT_START = set(  # Cannot start a line
    "、。，．！？）】」』〕〉》，．！？）］｝,.)!?;:…‥ー"
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
)
KINSOKU_NOT_AT_END = set("（【「『〔〈《（［｛([")  # Cannot end a line


def _split_with_cjk_awareness(
    text: str, detach_trailing_ellipsis: bool = True
) -> List[str]:
    """Split text into tokens. Each CJK char is a token; kinsoku rules apply.

    Hangul (Korean) is excluded from per-character splitting because Korean
    uses spaces between words — syllables accumulate into word-level tokens
    like Latin characters, preserving inter-word spacing.
    """
    tokens: List[str] = []
    current_token = ""

    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif is_cjk_character(char) and not _is_hangul_character(char):
            if char in KINSOKU_NOT_AT_START:
                if current_token:
                    current_token += char
                elif tokens:
                    tokens[-1] += char
                else:
                    current_token = char
            elif char in KINSOKU_NOT_AT_END:
                if current_token:
                    tokens.append(current_token)
                current_token = char
            else:
                if current_token:
                    if current_token[-1] in KINSOKU_NOT_AT_END:
                        current_token += char
                        tokens.append(current_token)
                        current_token = ""
                    else:
                        tokens.append(current_token)
                        current_token = ""
                        tokens.append(char)
                else:
                    tokens.append(char)
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    if detach_trailing_ellipsis:
        # Separate trailing ellipsis to allow wrapping
        final_tokens = []
        ellipsis_re = re.compile(r"^(.*?)((\.{2,})[\)\]\}\u2019\u201D\'\"]*)$")
        for t in tokens:
            m = ellipsis_re.match(t)
            if m and m.group(1):
                final_tokens.append(m.group(1))
                final_tokens.append(m.group(2))
            else:
                final_tokens.append(t)
        return final_tokens

    return tokens


def tokenize_styled_text(
    text: str, detach_trailing_ellipsis: bool = True
) -> List[Tuple[str, bool]]:
    """
    Tokenizes text into atomic units for wrapping where styled blocks are
    preserved as single, unbreakable tokens.

    Returns: List[Tuple[str, bool]] where each tuple is (token_text, is_styled).
    - Styled tokens are split into per-word tokens (CJK-aware), each wrapped
      with the same markers, to allow wrapping at word/character boundaries
      while preserving style.
    - Plain text outside markers is split with CJK awareness into word/character tokens.
    """
    tokens: List[Tuple[str, bool]] = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            preceding = text[last_end:start]
            for w in _split_with_cjk_awareness(preceding, detach_trailing_ellipsis):
                tokens.append((w, False))

        marker = match.group(1)
        content = match.group(2)
        if content:
            for w in _split_with_cjk_awareness(content, detach_trailing_ellipsis):
                tokens.append((f"{marker}{w}{marker}", True))

        last_end = end

    if last_end < len(text):
        trailing = text[last_end:]
        for w in _split_with_cjk_awareness(trailing, detach_trailing_ellipsis):
            tokens.append((w, False))

    return tokens


def try_hyphenate_word(
    word_str: str,
    min_word_length: int,
    width_test_func: Callable[[str], bool],
) -> Optional[List[str]]:
    """
    Attempts to split a word into two parts with a hyphen such that each part passes the width test.

    This is a generic hyphenation function that doesn't know about fonts or rendering.
    It uses a callback function to test if each part fits.

    Args:
        word_str: The word to potentially hyphenate
        min_word_length: Minimum word length to attempt hyphenation
        width_test_func: Function that takes a string and returns True if it fits

    Returns:
        List of two strings (the split parts) if successful, None otherwise
    """
    if any(0x0600 <= ord(c) <= 0x06FF or 0x0590 <= ord(c) <= 0x05FF for c in word_str):
        return None

    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", word_str)
    if not match:
        return None

    leading_punc, core_word, trailing_punc = match.groups()

    if len(core_word) < min_word_length:
        return None

    def _split_with_single_hyphen(base: str, idx: int) -> Tuple[str, str]:
        ch_before = base[idx - 1] if idx > 0 else ""
        ch_at = base[idx] if idx < len(base) else ""
        if ch_at == "-":
            left = base[: idx + 1]
            right = base[idx + 1 :]
        elif ch_before == "-":
            left = base[:idx]
            right = base[idx:]
        else:
            left = base[:idx] + "-"
            right = base[idx:]
        if left.endswith("-") and right.startswith("-"):
            right = right[1:]
        return left, right

    # Try splitting at existing hyphens first
    if "-" in core_word:
        hyphen_positions = [i for i, ch in enumerate(core_word) if ch == "-"]
        mid = len(core_word) // 2
        hyphen_positions.sort(key=lambda i: abs(i - mid))
        for pos in hyphen_positions:
            if pos <= 0 or pos >= len(core_word) - 1:
                continue
            left_part = core_word[: pos + 1]
            right_part = core_word[pos + 1 :]

            final_left_part = leading_punc + left_part
            final_right_part = right_part + trailing_punc

            if width_test_func(final_left_part) and width_test_func(final_right_part):
                return [final_left_part, final_right_part]

    # Try splitting at various positions
    mid = len(core_word) // 2
    candidate_indices: List[int] = []
    max_d = max(mid, len(core_word) - mid)
    for d in range(0, max_d):
        left_idx = mid - d
        right_idx = mid + d
        if 2 <= left_idx < len(core_word) - 2:
            candidate_indices.append(left_idx)
        if 2 <= right_idx < len(core_word) - 2 and right_idx != left_idx:
            candidate_indices.append(right_idx)

    for idx in candidate_indices:
        left_part, right_part = _split_with_single_hyphen(core_word, idx)

        final_left_part = leading_punc + left_part
        final_right_part = right_part + trailing_punc

        if width_test_func(final_left_part) and width_test_func(final_right_part):
            return [final_left_part, final_right_part]

    return None


def _is_cjk_token(token: str) -> bool:
    """Check if token consists entirely of spaceless CJK (Chinese/Japanese, not Hangul)."""
    match = STYLE_PATTERN.match(token)
    content = match.group(2) if match else token
    return len(content) > 0 and all(
        is_cjk_character(c) and not _is_hangul_character(c) for c in content
    )


def _needs_space_between(
    left_token: str, right_token: str, detach_trailing_ellipsis: bool = True
) -> bool:
    """No space needed between adjacent CJK tokens or before separated punctuation."""
    if _is_cjk_token(left_token) and _is_cjk_token(right_token):
        return False

    if detach_trailing_ellipsis:
        match = STYLE_PATTERN.match(right_token)
        r_content = match.group(2) if match else right_token
        # No space before detached ellipsis/punctuation chunks
        if re.match(r"^(\.{2,})[\)\]\}\u2019\u201D\'\"]*$", r_content):
            return False

    return True


def _join_tokens_smart(tokens: List[str], detach_trailing_ellipsis: bool = True) -> str:
    """Join tokens with smart spacing (no space between adjacent CJK tokens)."""
    if not tokens:
        return ""
    result = tokens[0]
    for i in range(1, len(tokens)):
        if _needs_space_between(tokens[i - 1], tokens[i], detach_trailing_ellipsis):
            result += " " + tokens[i]
        else:
            result += tokens[i]
    return result


def find_optimal_breaks_dp(
    tokens: List[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    badness_exponent: float = 3.0,
    hyphen_penalty: float = 1000.0,
    detach_trailing_ellipsis: bool = True,
) -> Optional[List[str]]:
    """
    Pragmatic Knuth-Plass style DP to find globally optimal line breaks.

    This is a pure algorithm that doesn't know about fonts or rendering.
    It uses callback functions to get widths.

    Args:
        tokens: List of word tokens
        max_width: Maximum allowed line width
        word_width_func: Function that takes a word and returns its width
        space_width: Width of a space character
        badness_exponent: Exponent for badness calculation (higher = prefer tighter lines)
        hyphen_penalty: Penalty for lines ending with hyphens

    Returns:
        List of lines (strings) if successful, None if impossible to fit
    """
    try:
        if not tokens:
            return []

        # Calculate widths for all tokens
        token_w: List[float] = [word_width_func(t) for t in tokens]

        N = len(tokens)
        min_cost: List[float] = [float("inf")] * (N + 1)
        path: List[int] = [0] * (N + 1)
        min_cost[0] = 0.0

        for i in range(1, N + 1):
            line_width = 0.0
            for j in range(i - 1, -1, -1):
                # Add space only if needed between this token and the previous one on the line
                if j < i - 1:
                    # Check if we need space between tokens[j] and tokens[j+1]
                    if _needs_space_between(
                        tokens[j], tokens[j + 1], detach_trailing_ellipsis
                    ):
                        line_width += space_width
                line_width += token_w[j]

                if line_width > max_width:
                    break

                slack = max_width - line_width
                badness = pow(slack, badness_exponent)

                # Add hyphen penalty if line ends with hyphen (support styled markers)
                last_token = tokens[i - 1] if i > 0 else ""
                ends_with_hyphen = last_token.endswith("-")
                if not ends_with_hyphen:
                    styled_match = STYLE_PATTERN.match(last_token)
                    if styled_match:
                        ends_with_hyphen = styled_match.group(2).endswith("-")
                if ends_with_hyphen:
                    badness += hyphen_penalty

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        lines: List[str] = []
        current_break = N
        while current_break > 0:
            prev_break = path[current_break]
            line = _join_tokens_smart(
                tokens[prev_break:current_break], detach_trailing_ellipsis
            )
            lines.insert(0, line)
            current_break = prev_break

        return lines

    except Exception:
        return None


def _strip_style(word: str) -> str:
    """Strip style markers (***, **, *) from a token, returning bare content."""
    m = STYLE_PATTERN.match(word)
    return m.group(2) if m else word


def _estimated_word_width(word: str, font_size: int, avg_char_width: float) -> float:
    """Rough width estimate using character count, not HarfBuzz."""
    bare = _strip_style(word)
    return len(bare) * avg_char_width


def _typeR_circular(
    tokens: List[str],
    char_lens: List[int],
    n_words: int,
    max_width: float,
    font_size: int,
    space_width: float,
    detach_trailing_ellipsis: bool,
) -> Optional[List[str]]:
    """Circular/square bubble → diamond shape: middle lines widest, edges shortest."""
    CHAR_WIDTH_FACTOR = 0.55
    avg_char_width = font_size * CHAR_WIDTH_FACTOR
    word_est_widths = [cl * avg_char_width for cl in char_lens]

    if n_words <= 5:
        n_lines = 2
    elif n_words <= 8:
        n_lines = 3
    elif n_words <= 12:
        n_lines = 4
    elif n_words <= 15:
        n_lines = 5
    elif n_words <= 25:
        n_lines = 6
    else:
        n_lines = 7
    n_lines = min(n_lines, n_words)
    if n_lines < 3:
        return None

    total_est = word_est_widths[0]
    for i in range(1, n_words):
        if _needs_space_between(tokens[i - 1], tokens[i], detach_trailing_ellipsis):
            total_est += space_width
        total_est += word_est_widths[i]

    diamond_ratios = {
        3: [0.78, 1.00, 0.78],
        4: [0.78, 1.00, 1.00, 0.78],
        5: [0.62, 0.82, 1.00, 0.82, 0.62],
        6: [0.60, 0.78, 1.00, 1.00, 0.78, 0.60],
    }
    if n_lines in diamond_ratios:
        ratios = diamond_ratios[n_lines]
    else:
        center = (n_lines - 1) / 2.0
        ratios = [
            max(0.58, 1.00 - abs(i - center) / max(center, 1) * 0.40)
            for i in range(n_lines)
        ]

    avg_width = total_est / n_lines
    ref_width = min(max_width, avg_width * 1.35)
    targets = [min(ref_width * r, max_width) for r in ratios]

    return _typeR_dp(tokens, word_est_widths, n_words, n_lines, targets, max_width, space_width, detach_trailing_ellipsis)


def _typeR_wide(
    tokens: List[str],
    char_lens: List[int],
    n_words: int,
    max_width: float,
    font_size: int,
    space_width: float,
    detach_trailing_ellipsis: bool,
) -> Optional[List[str]]:
    """Wide bubble (AR > 2) → fewer lines, each filling the full width."""
    CHAR_WIDTH_FACTOR = 0.55
    avg_char_width = font_size * CHAR_WIDTH_FACTOR
    word_est_widths = [cl * avg_char_width for cl in char_lens]

    if n_words <= 3:
        n_lines = 1
    elif n_words <= 7:
        n_lines = 2
    elif n_words <= 12:
        n_lines = 3
    elif n_words <= 18:
        n_lines = 4
    else:
        n_lines = 5
    n_lines = min(n_lines, n_words)
    if n_lines < 2:
        return None

    # All lines fill the full width equally (rectangle shape)
    targets = [max_width] * n_lines

    return _typeR_dp(tokens, word_est_widths, n_words, n_lines, targets, max_width, space_width, detach_trailing_ellipsis)


def _typeR_tall(
    tokens: List[str],
    char_lens: List[int],
    n_words: int,
    max_width: float,
    font_size: int,
    space_width: float,
    detach_trailing_ellipsis: bool,
) -> Optional[List[str]]:
    """Tall bubble (AR < 0.5) → more lines, narrower columns."""
    CHAR_WIDTH_FACTOR = 0.55
    avg_char_width = font_size * CHAR_WIDTH_FACTOR
    word_est_widths = [cl * avg_char_width for cl in char_lens]

    if n_words <= 5:
        n_lines = 2
    elif n_words <= 8:
        n_lines = 3
    elif n_words <= 11:
        n_lines = 4
    elif n_words <= 15:
        n_lines = 5
    elif n_words <= 20:
        n_lines = 6
    elif n_words <= 26:
        n_lines = 7
    elif n_words <= 33:
        n_lines = 8
    else:
        n_lines = 9
    n_lines = min(n_lines, n_words)
    if n_lines < 3:
        return None

    # Narrow diamond: peaks are narrower than max_width
    diamond_ratios = {
        3: [0.65, 0.85, 0.65],
        4: [0.60, 0.85, 0.85, 0.60],
        5: [0.55, 0.75, 0.85, 0.75, 0.55],
        6: [0.50, 0.70, 0.85, 0.85, 0.70, 0.50],
    }
    if n_lines in diamond_ratios:
        ratios = diamond_ratios[n_lines]
    else:
        center = (n_lines - 1) / 2.0
        ratios = [
            max(0.50, 1.00 - abs(i - center) / max(center, 1) * 0.50)
            for i in range(n_lines)
        ]

    targets = [max_width * r for r in ratios]

    return _typeR_dp(tokens, word_est_widths, n_words, n_lines, targets, max_width, space_width, detach_trailing_ellipsis)


def _typeR_dp(
    tokens: List[str],
    word_est_widths: List[float],
    n_words: int,
    n_lines: int,
    targets: List[float],
    max_width: float,
    space_width: float,
    detach_trailing_ellipsis: bool,
) -> Optional[List[str]]:
    """DP: distribute words across lines to minimize squared deviation from targets."""
    # Prefix sums of estimated widths (including inter-word spaces)
    prefix: List[float] = [0.0]
    for i in range(n_words):
        base = prefix[-1]
        if i > 0 and _needs_space_between(tokens[i - 1], tokens[i], detach_trailing_ellipsis):
            base += space_width
        prefix.append(base + word_est_widths[i])

    INF = float("inf")
    dp = [[INF] * (n_lines + 1) for _ in range(n_words + 1)]
    prev = [[0] * (n_lines + 1) for _ in range(n_words + 1)]
    dp[0][0] = 0.0

    for k in range(1, n_lines + 1):
        target = targets[k - 1]
        min_s = k
        max_s = n_words - (n_lines - k)
        for s in range(min_s, max_s + 1):
            best_g = -1
            best_cost = INF
            for g in range(k - 1, s):
                if dp[g][k - 1] >= INF:
                    continue
                line_width = prefix[s] - prefix[g]
                if line_width > max_width:
                    continue
                diff = line_width - target
                cost = dp[g][k - 1] + diff * diff
                if cost < best_cost:
                    best_cost = cost
                    best_g = g
            if best_g >= 0:
                dp[s][k] = best_cost
                prev[s][k] = best_g

    if dp[n_words][n_lines] >= INF:
        return None

    breaks: List[int] = []
    curr = n_words
    for k in range(n_lines, 0, -1):
        breaks.append(prev[curr][k])
        curr = prev[curr][k]
    breaks.reverse()
    breaks.append(n_words)

    lines_result: List[str] = []
    for i in range(n_lines):
        line_tokens = tokens[breaks[i]:breaks[i + 1]]
        lines_result.append(
            _join_tokens_smart(line_tokens, detach_trailing_ellipsis)
        )

    return lines_result


def typeR_diamond_break(
    tokens: List[str],
    max_width: float,
    max_height: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    detach_trailing_ellipsis: bool = True,
    font_size: int = 12,
) -> Optional[List[str]]:
    """TypeR-inspired layout: aspect-ratio strategy selection + char-count DP.

    Uses character count (not HarfBuzz) for line-breaking decisions, with
    rough estimated widths (``font_size × 0.55 × char_count``).  The actual
    rendered widths come from HarfBuzz at draw time — this function only
    decides *where* lines break.

    Strategy selection by bubble aspect ratio (width/height):
        * **wide** (>2):   fewer lines, rectangle fill.
        * **tall** (<0.5): more lines, narrow diamond.
        * **circular**:    diamond shape (middle wide, edges short).

    Falls back to ``find_optimal_breaks_dp`` + ``reflow_diamond`` on failure.
    """
    n = len(tokens)
    if n <= 1:
        return None

    # Pre-compute character lengths (stripping style markers)
    char_lens = [len(_strip_style(t)) for t in tokens]

    aspect_ratio = max_width / max(max_height, 1.0)

    if aspect_ratio > 2.0:
        return _typeR_wide(tokens, char_lens, n, max_width, font_size, space_width, detach_trailing_ellipsis)
    elif aspect_ratio < 0.5:
        return _typeR_tall(tokens, char_lens, n, max_width, font_size, space_width, detach_trailing_ellipsis)
    else:
        return _typeR_circular(tokens, char_lens, n, max_width, font_size, space_width, detach_trailing_ellipsis)


def reflow_diamond(
    lines: List[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
) -> List[str]:
    """
    Post-processes line breaks to create a diamond/pyramid layout:
    middle lines are the widest, top and bottom edges are progressively shorter.

    Key insight: Reference = max of ACTUAL line widths, NOT bubble width.
    This matches the JSX implementation and produces better results:
    - If all lines are naturally equal, no unnecessary stretching
    - Only lines shorter than their diamond target get adjusted
    """
    if len(lines) < 3:
        return lines

    def measure_line(line: str) -> float:
        words = line.split(" ")
        if not words or words == [""]:
            return 0.0
        total = sum(word_width_func(w) for w in words if w)
        total += space_width * max(0, len([w for w in words if w]) - 1)
        return total

    n = len(lines)
    center = (n - 1) / 2.0

    actual_widths = [measure_line(line) for line in lines]
    max_actual_width = max(actual_widths) if actual_widths else max_width
    if max_actual_width <= 0:
        return lines

    import math

    diamond_ratios = {
        1: [1.00],
        2: [1.00, 1.00],
        3: [0.78, 1.00, 0.78],
        4: [0.78, 1.00, 1.00, 0.78],
        5: [0.62, 0.82, 1.00, 0.82, 0.62],
        6: [0.60, 0.78, 1.00, 1.00, 0.78, 0.60],
    }

    if n in diamond_ratios:
        target_ratios = diamond_ratios[n]
    else:
        target_ratios = [
            max(0.58, 1.00 - abs(i - center) / max(center, 1) * 0.40)
            for i in range(n)
        ]

    targets = [max_actual_width * r for r in target_ratios]

    result = list(lines)
    for _pass in range(4):
        changed = False
        for i in range(len(result) - 1):
            words_i = [w for w in result[i].split(" ") if w]
            words_next = [w for w in result[i + 1].split(" ") if w]
            if len(words_i) < 2:
                continue
            current_w_i = measure_line(result[i])
            current_w_next = measure_line(result[i + 1])
            target_i = targets[i]
            target_next = targets[i + 1]
            candidate_i = " ".join(words_i[:-1])
            candidate_next = words_i[-1] + " " + result[i + 1].strip()
            new_w_i = measure_line(candidate_i)
            new_w_next = measure_line(candidate_next)
            if new_w_next <= max_width * 1.02:
                old_err = abs(current_w_i - target_i) + abs(current_w_next - target_next)
                new_err = abs(new_w_i - target_i) + abs(new_w_next - target_next)
                if new_err < old_err * 0.92:
                    result[i] = candidate_i
                    result[i + 1] = candidate_next
                    changed = True
        if not changed:
            break
    return result


_ALEF_CHARS = set("آآأإٱ")


def _fix_lam_alef_fallback(word: str) -> str:
    if len(word) <= 2:
        return word
    result = []
    for i, char in enumerate(word):
        result.append(char)
        if i > 0 and char in _ALEF_CHARS and i + 1 < len(word) and word[i + 1] == "\u0644":
            result.append("\u200C")
    return "".join(result)


_LAM_CHARS = r"[\u0644\uFEDD\uFEDE\uFEDF\uFEE0]"
_ALEF_CHARS_RX = r"[\u0622\u0623\u0625\u0627\u0671\uFE81\uFE82\uFE83\uFE84\uFE87\uFE88\uFE8D\uFE8E\uFB50\uFB51]"


def preprocess_arabic_text(text: str) -> str:
    """
    Passthrough for Arabic text. HarfBuzz with the font's GSUB features
    (init/medi/fina/isol via calt) handles Arabic positional form selection
    and BiDi reordering natively.  Only the Lam-Alef ZWNJ fix is applied
    to prevent unwanted ligation.
    """
    return re.sub(f"({_LAM_CHARS})({_ALEF_CHARS_RX})", r"\1" + "\u200C" + r"\2", text)
