"""
Utility functions for PDF outline extraction.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from config import *


def calculate_font_statistics(pages: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate font size statistics across all pages.

    Args:
        pages: List of page data from PDF extraction

    Returns:
        Dictionary with font statistics
    """
    font_sizes = []
    font_families = {}

    for page in pages:
        for span in page["text_with_fonts"]:
            text = span["text"].strip()
            if text and len(text) > 2:
                font_sizes.append(span["size"])
                font_family = span["font"]
                font_families[font_family] = font_families.get(font_family, 0) + 1

    if not font_sizes:
        return {
            "avg_font_size": 12,
            "max_font_size": 12,
            "min_font_size": 12,
            "common_font": "",
            "large_threshold": 18,
            "medium_threshold": 14.4,
            "small_threshold": 13.2,
        }

    avg_font_size = sum(font_sizes) / len(font_sizes)
    max_font_size = max(font_sizes)
    min_font_size = min(font_sizes)
    common_font = max(font_families, key=font_families.get) if font_families else ""

    return {
        "avg_font_size": avg_font_size,
        "max_font_size": max_font_size,
        "min_font_size": min_font_size,
        "common_font": common_font,
        "large_threshold": avg_font_size * FONT_SIZE_THRESHOLDS["large"],
        "medium_threshold": avg_font_size * FONT_SIZE_THRESHOLDS["medium"],
        "small_threshold": avg_font_size * FONT_SIZE_THRESHOLDS["small"],
    }


def group_text_into_lines(spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group text spans into lines based on y-coordinates.

    Args:
        spans: List of text spans from a page

    Returns:
        List of lines, where each line is a list of spans
    """
    lines = []
    current_line = []
    last_y = None
    tolerance = PROCESSING_SETTINGS["line_tolerance"]

    # Sort spans by y-coordinate, then x-coordinate
    sorted_spans = sorted(spans, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    for span in sorted_spans:
        text = span["text"].strip()
        if not text:
            continue

        y_pos = span["bbox"][1]  # top y coordinate

        if last_y is None or abs(y_pos - last_y) <= tolerance:
            current_line.append(span)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [span]

        last_y = y_pos

    if current_line:
        lines.append(current_line)

    return lines


def is_likely_body_text(text: str) -> bool:
    """
    Check if text is likely body text (not a heading).

    Args:
        text: Text to analyze

    Returns:
        True if likely body text, False otherwise
    """
    if len(text) > TEXT_LENGTH["max_heading_length"]:
        return True

    words = text.lower().split()
    if len(words) <= 5:
        return False

    # Check ratio of common words
    common_count = sum(1 for word in words if word in COMMON_BODY_WORDS)
    common_ratio = common_count / len(words)

    return common_ratio > COMMON_WORDS_RATIO_THRESHOLD


def get_pattern_score_and_level(text: str) -> Tuple[int, str]:
    """
    Get pattern matching score and suggested level for text.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, suggested_level)
    """
    for pattern, weight, level in HEADING_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return weight, level

    return 0, ""


def calculate_heading_score(
    line: List[Dict[str, Any]],
    line_text: str,
    font_stats: Dict[str, float],
    page_num: int,
) -> Dict[str, Any]:
    """
    Calculate heading score for a line of text.

    Args:
        line: List of text spans in the line
        line_text: Combined text of the line
        font_stats: Font statistics for the document
        page_num: Current page number

    Returns:
        Dictionary with score and analysis results
    """
    if not line or not line_text:
        return {"score": 0, "level": None}

    # Calculate line characteristics
    avg_line_size = sum(span["size"] for span in line) / len(line)
    max_line_size = max(span["size"] for span in line)

    # Check formatting flags
    is_bold = any(span["flags"] & 16 for span in line)
    is_italic = any(span["flags"] & 2 for span in line)

    # Check font consistency
    line_fonts = set(span["font"] for span in line)
    is_different_font = (
        len(line_fonts) == 1 and list(line_fonts)[0] != font_stats["common_font"]
    )

    # Position analysis
    y_position = line[0]["bbox"][1]
    x_position = line[0]["bbox"][0]

    # Initialize score
    score = 0
    level_hints = []

    # 1. Font size scoring
    if avg_line_size >= font_stats["large_threshold"]:
        score += SCORING_WEIGHTS["font_size_large"]
        level_hints.append("H1")
    elif avg_line_size >= font_stats["medium_threshold"]:
        score += SCORING_WEIGHTS["font_size_medium"]
        level_hints.append("H2")
    elif avg_line_size >= font_stats["small_threshold"]:
        score += SCORING_WEIGHTS["font_size_small"]
        level_hints.append("H3")

    # 2. Font formatting scoring
    if is_bold:
        score += SCORING_WEIGHTS["bold_formatting"]
    if is_italic:
        score += SCORING_WEIGHTS["italic_formatting"]
    if is_different_font:
        score += SCORING_WEIGHTS["different_font"]

    # 3. Pattern matching scoring
    pattern_score, pattern_level = get_pattern_score_and_level(line_text)
    score += pattern_score
    if pattern_level:
        level_hints.append(pattern_level)

    # 4. Position scoring
    page_height = PAGE_DIMENSIONS["height"]
    if y_position < page_height * POSITION_THRESHOLDS["top_section"]:
        score += SCORING_WEIGHTS["top_position"]
    elif y_position < page_height * POSITION_THRESHOLDS["upper_section"]:
        score += SCORING_WEIGHTS["upper_position"]

    if x_position < POSITION_THRESHOLDS["left_margin"]:
        score += SCORING_WEIGHTS["left_margin"]

    # 5. Capitalization scoring
    if line_text.isupper() and len(line_text) < 50:
        score += SCORING_WEIGHTS["all_caps"]
        level_hints.append("H1")
    elif line_text.istitle():
        score += SCORING_WEIGHTS["title_case"]

    # 6. Length scoring
    word_count = len(line_text.split())
    if word_count <= TEXT_LENGTH["short_text_words"]:
        score += SCORING_WEIGHTS["short_text"]
    elif word_count <= TEXT_LENGTH["medium_text_words"]:
        score += SCORING_WEIGHTS["medium_text"]
    elif word_count > TEXT_LENGTH["long_text_words"]:
        score -= 1

    # 7. Ending punctuation scoring
    if line_text.endswith(":"):
        score += SCORING_WEIGHTS["ends_with_colon"]

    # Determine level
    level = determine_heading_level(level_hints, avg_line_size, font_stats, line_text)

    return {
        "score": score,
        "level": level,
        "font_size": avg_line_size,
        "is_bold": is_bold,
        "y_position": y_position,
        "level_hints": level_hints,
    }


def determine_heading_level(
    level_hints: List[str], avg_size: float, font_stats: Dict[str, float], text: str
) -> str:
    """
    Determine the heading level based on various hints.

    Args:
        level_hints: List of level suggestions from various analyses
        avg_size: Average font size of the line
        font_stats: Font statistics for the document
        text: The text content

    Returns:
        Heading level (H1, H2, or H3)
    """
    # Count level hints
    level_counts = {"H1": 0, "H2": 0, "H3": 0}
    for hint in level_hints:
        if hint in level_counts:
            level_counts[hint] += 1

    # Find the most suggested level
    suggested_level = max(level_counts, key=level_counts.get)

    # If no clear winner, use font size
    if level_counts[suggested_level] == 0:
        if avg_size >= font_stats["large_threshold"]:
            return "H1"
        elif avg_size >= font_stats["medium_threshold"]:
            return "H2"
        else:
            return "H3"

    # Use pattern-based fallback
    if level_counts[suggested_level] == 0:
        if re.match(r"^\d+\.\s+", text):
            return "H1"
        elif re.match(r"^\d+\.\d+\s+", text):
            return "H2"
        elif re.match(r"^\d+\.\d+\.\d+\s+", text):
            return "H3"

    return suggested_level


def is_potential_title(
    text: str,
    score: int,
    avg_size: float,
    font_stats: Dict[str, float],
    page_num: int,
    is_bold: bool,
    y_position: float,
) -> bool:
    """
    Check if text is a potential document title.

    Args:
        text: Text to check
        score: Current heading score
        avg_size: Average font size of the text
        font_stats: Font statistics
        page_num: Page number
        is_bold: Whether text is bold
        y_position: Y position on page

    Returns:
        True if potential title, False otherwise
    """
    if page_num != 1:
        return False

    if score < PROCESSING_SETTINGS["title_detection_threshold"]:
        return False

    page_height = PAGE_DIMENSIONS["height"]
    if y_position > page_height * POSITION_THRESHOLDS["upper_section"]:
        return False

    # Check font size criteria
    large_font = (
        avg_size
        >= font_stats["avg_font_size"]
        * PROCESSING_SETTINGS["title_font_size_multiplier"]
    )
    bold_medium_font = (
        is_bold
        and avg_size
        >= font_stats["avg_font_size"]
        * PROCESSING_SETTINGS["title_bold_size_multiplier"]
    )

    return large_font or bold_medium_font


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def filter_similar_headings(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out similar or duplicate headings.

    Args:
        headings: List of heading dictionaries

    Returns:
        Filtered list of headings
    """
    if not PROCESSING_SETTINGS["enable_similarity_filtering"]:
        return headings

    filtered = []
    seen_texts = set()

    for heading in headings:
        text = heading["text"].strip()
        text_lower = text.lower()

        # Skip empty or very short headings
        if len(text) < TEXT_LENGTH["min_heading_length"]:
            continue

        # Skip exact duplicates
        if text_lower in seen_texts:
            continue

        # Check for high similarity with existing headings
        is_similar = False
        for seen_text in seen_texts:
            # Check substring relationships
            if text_lower in seen_text or seen_text in text_lower:
                char_diff = abs(len(text_lower) - len(seen_text))
                if char_diff < SIMILARITY_THRESHOLDS["character_difference"]:
                    is_similar = True
                    break

            # Check semantic similarity
            similarity = calculate_text_similarity(text_lower, seen_text)
            if similarity > SIMILARITY_THRESHOLDS["high_similarity"]:
                is_similar = True
                break

        if not is_similar:
            seen_texts.add(text_lower)
            filtered.append(heading)

    return filtered
