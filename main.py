import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import regex as re

# Import our utility modules
from utils import (
    calculate_font_statistics,
    group_text_into_lines,
    is_likely_body_text,
    calculate_heading_score,
    is_potential_title,
    filter_similar_headings,
)
from config import MIN_HEADING_SCORE, PROCESSING_SETTINGS


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Module 1: PDF Text and Metadata Extraction
def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text content and metadata from PDF with positional information.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing text content per page and metadata
    """
    try:
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text blocks with positional information
            text_blocks = page.get_text("dict")

            # Extract plain text for fallback
            plain_text = page.get_text()

            # Extract text with font information
            text_with_fonts = []
            for block in text_blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_with_fonts.append(
                                {
                                    "text": span["text"],
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "bbox": span["bbox"],
                                    "color": span.get("color", 0),
                                }
                            )

            pages_data.append(
                {
                    "page_number": page_num + 1,
                    "plain_text": plain_text,
                    "text_with_fonts": text_with_fonts,
                    "text_blocks": text_blocks,
                }
            )

        doc.close()

        metadata = doc.metadata if hasattr(doc, "metadata") else {}

        return {
            "pages": pages_data,
            "total_pages": len(pages_data),
            "metadata": {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
            },
        }

    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise


# Module 2: Heading and Title Detection
def identify_headings(pdf_content: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Identify document title and headings using multiple heuristics.

    Args:
        pdf_content: Output from extract_pdf_content

    Returns:
        Tuple of (title, list of headings)
    """
    pages = pdf_content["pages"]
    all_headings = []
    document_title = ""

    # First, try to get title from metadata
    if pdf_content["metadata"]["title"]:
        document_title = pdf_content["metadata"]["title"].strip()

    # Calculate font statistics across the document
    font_stats = calculate_font_statistics(pages)

    # Process each page
    for page in pages:
        page_num = page["page_number"]

        # Group text into lines
        lines = group_text_into_lines(page["text_with_fonts"])

        # Analyze each line for heading characteristics
        for line in lines:
            if not line:
                continue

            line_text = " ".join(span["text"] for span in line).strip()

            # Skip invalid candidates
            if not line_text or is_likely_body_text(line_text):
                continue

            # Calculate heading score and characteristics
            analysis = calculate_heading_score(line, line_text, font_stats, page_num)

            # Check if this could be the document title
            if not document_title and is_potential_title(
                line_text,
                analysis["score"],
                analysis["font_size"],
                font_stats,
                page_num,
                analysis["is_bold"],
                analysis["y_position"],
            ):
                document_title = line_text
                continue

            # Add to headings if score is high enough
            if analysis["score"] >= MIN_HEADING_SCORE:
                all_headings.append(
                    {
                        "text": line_text,
                        "level": analysis["level"],
                        "page": page_num,
                        "score": analysis["score"],
                        "font_size": analysis["font_size"],
                        "is_bold": analysis["is_bold"],
                        "y_position": analysis["y_position"],
                    }
                )

    # Post-process headings
    processed_headings = post_process_headings(all_headings)

    # Final title extraction if still not found
    if not document_title and processed_headings:
        first_heading = processed_headings[0]
        if first_heading["page"] == 1:
            document_title = first_heading["text"]
            processed_headings = processed_headings[1:]

    return document_title, processed_headings


def post_process_headings(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process headings to ensure quality and consistency.

    Args:
        headings: List of raw heading dictionaries

    Returns:
        List of processed heading dictionaries
    """
    if not headings:
        return []

    # Sort by page number, then by y-position (top to bottom), then by score
    sorted_headings = sorted(
        headings, key=lambda x: (x["page"], x.get("y_position", 0), -x["score"])
    )

    # Filter similar headings
    filtered_headings = filter_similar_headings(sorted_headings)

    # Ensure hierarchical consistency if enabled
    if PROCESSING_SETTINGS["enable_hierarchical_adjustment"]:
        adjusted_headings = ensure_hierarchical_consistency(filtered_headings)
    else:
        adjusted_headings = filtered_headings

    # Return only the required fields
    return [
        {"text": heading["text"], "level": heading["level"], "page": heading["page"]}
        for heading in adjusted_headings
    ]


def ensure_hierarchical_consistency(
    headings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure that heading levels follow a logical hierarchy (H1 -> H2 -> H3).

    Args:
        headings: List of heading dictionaries

    Returns:
        List of headings with adjusted levels
    """
    if not headings:
        return []

    adjusted = []
    last_level = None
    level_counts = {"H1": 0, "H2": 0, "H3": 0}

    for heading in headings:
        current_level = heading["level"]

        # Track level usage
        level_counts[current_level] += 1

        # Adjust level based on hierarchy rules
        if last_level is None:
            # First heading should generally be H1
            if current_level in ["H2", "H3"] and level_counts["H1"] == 0:
                current_level = "H1"
        else:
            # Ensure we don't skip levels inappropriately
            if current_level == "H3" and level_counts["H2"] == 0:
                current_level = "H2"

        # Update the heading with the adjusted level
        adjusted_heading = heading.copy()
        adjusted_heading["level"] = current_level
        adjusted.append(adjusted_heading)

        last_level = current_level

    return adjusted


# Module 3: Outline Formatting and JSON Output
def format_outline_to_json(
    title: str, headings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format the extracted title and headings into the required JSON structure.

    Args:
        title: Document title
        headings: List of heading dictionaries

    Returns:
        Formatted JSON-compatible dictionary
    """
    return {"title": title or "Untitled Document", "outline": headings}


def save_json_output(data: Dict[str, Any], output_path: str) -> None:
    """
    Save the formatted data to a JSON file.

    Args:
        data: Dictionary to save as JSON
        output_path: Path where to save the JSON file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON output: {e}")
        raise


def process_single_pdf(pdf_path: str, output_path: str) -> None:
    """
    Process a single PDF file and generate JSON output.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path for output JSON file
    """
    try:
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract PDF content
        pdf_content = extract_pdf_content(pdf_path)
        logger.info(f"Extracted content from {pdf_content['total_pages']} pages")

        # Identify headings and title
        title, headings = identify_headings(pdf_content)
        logger.info(f"Identified title: '{title}' and {len(headings)} headings")

        # Format to JSON structure
        json_data = format_outline_to_json(title, headings)

        # Save output
        save_json_output(json_data, output_path)

        logger.info(f"Successfully processed: {pdf_path}")

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise


def main():
    """
    Main execution logic - process all PDFs in /app/input directory.
    """
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF file
    processed_count = 0
    failed_count = 0

    for pdf_file in pdf_files:
        try:
            # Generate output filename
            json_filename = pdf_file.stem + ".json"
            output_path = output_dir / json_filename

            # Process the PDF
            process_single_pdf(str(pdf_file), str(output_path))
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
            failed_count += 1
            continue

    logger.info(
        f"Processing complete! Processed: {processed_count}, Failed: {failed_count}"
    )


if __name__ == "__main__":
    main()
