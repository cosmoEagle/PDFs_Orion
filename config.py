# Configuration for PDF Outline Extractor
# Adjust these values to fine-tune heading detection

# Font size thresholds (multipliers of average font size)
FONT_SIZE_THRESHOLDS = {
    "large": 1.5,  # For H1 detection
    "medium": 1.2,  # For H2 detection
    "small": 1.1,  # For H3 detection
}

# Scoring weights for different heading indicators
SCORING_WEIGHTS = {
    "font_size_large": 4,
    "font_size_medium": 3,
    "font_size_small": 2,
    "bold_formatting": 3,
    "italic_formatting": 1,
    "different_font": 1,
    "pattern_match_high": 3,
    "pattern_match_medium": 2,
    "pattern_match_low": 1,
    "top_position": 2,
    "upper_position": 1,
    "left_margin": 1,
    "all_caps": 2,
    "title_case": 1,
    "short_text": 2,
    "medium_text": 1,
    "ends_with_colon": 1,
}

# Minimum score threshold for heading detection
MIN_HEADING_SCORE = 5

# Text length thresholds
TEXT_LENGTH = {
    "min_heading_length": 2,
    "max_heading_length": 150,
    "short_text_words": 5,
    "medium_text_words": 10,
    "long_text_words": 20,
}

# Position thresholds (as fraction of page height)
POSITION_THRESHOLDS = {
    "top_section": 0.15,  # Top 15% of page
    "upper_section": 0.3,  # Top 30% of page
    "left_margin": 50,  # Left margin in pixels
}

# Similarity thresholds for duplicate detection
SIMILARITY_THRESHOLDS = {
    "exact_match": 1.0,
    "high_similarity": 0.7,
    "character_difference": 5,
}

# Document structure patterns
HEADING_PATTERNS = [
    # Pattern, Weight, Suggested Level
    (r"^(?:chapter|section|part|appendix)\s+\d+", 3, "H1"),
    (r"^(?:chapter|section|part|appendix)\s+[ivxlcdm]+", 3, "H1"),
    (r"^\d+\.\s+", 3, "H1"),
    (r"^\d+\.\d+\s+", 2, "H2"),
    (r"^\d+\.\d+\.\d+\s+", 1, "H3"),
    (r"^[A-Z]\.\s+", 2, "H2"),
    (r"^[a-z]\)\s+", 1, "H3"),
    (r"^[IVX]+\.\s+", 3, "H1"),
    (r"^[A-Z][A-Z\s]{2,}$", 2, "H1"),
    (r"^[A-Z][a-z\s]+:$", 2, "H2"),
    (r"^summary$|^conclusion$|^introduction$|^abstract$", 2, "H1"),
    (r"^references$|^bibliography$|^index$", 1, "H1"),
]

# Common words that indicate body text (not headings)
COMMON_BODY_WORDS = {
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "as",
    "if",
    "when",
    "where",
    "why",
    "how",
    "who",
    "what",
    "which",
    "there",
    "here",
    "then",
    "than",
    "so",
    "very",
    "just",
    "now",
    "also",
    "only",
    "other",
    "such",
    "some",
    "more",
    "most",
    "much",
    "many",
    "few",
    "little",
    "less",
}

# Ratio of common words that indicates body text
COMMON_WORDS_RATIO_THRESHOLD = 0.6

# Page dimensions (standard letter size)
PAGE_DIMENSIONS = {"height": 792, "width": 612}

# Processing settings
PROCESSING_SETTINGS = {
    "line_tolerance": 3,  # Pixel tolerance for grouping text into lines
    "enable_hierarchical_adjustment": True,
    "enable_duplicate_removal": True,
    "enable_similarity_filtering": True,
    "title_detection_threshold": 4,
    "title_font_size_multiplier": 1.8,
    "title_bold_size_multiplier": 1.3,
}
