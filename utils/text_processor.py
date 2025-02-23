import re


class TextProcessor:
    """Class for text processing and cleaning"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text from PDF"""
        # Remove multiple spaces and newlines
        text = re.sub(r"\s+", " ", text)
        # Remove hyphenation at line breaks
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        # Remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Remove common PDF artifacts
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
        return text.strip()

    @staticmethod
    def is_title(line: str) -> bool:
        """Check if a line is likely a section title"""
        # Title characteristics
        patterns = [
            r"^\d+\.\s+[A-Z]",  # Numbered sections
            r"^[A-Z][a-zA-Z\s]{2,50}$",  # Capitalized words
            r"^(?:Introduction|Methodology|Results|Discussion|Conclusion|References)",  # Common section names
            r"^(?:Abstract|Keywords|Background|Related Work|Literature Review)",  # Front matter
            r"^(?:Methods|Experimental Setup|Materials|Procedure|Design)",  # Methods sections
            r"^(?:Analysis|Evaluation|Findings|Observations|Performance)",  # Results sections
            r"^(?:Implementation|System Design|Architecture|Algorithm)",  # Technical sections
            r"^(?:Discussion|Interpretation|Implications|Limitations)",  # Discussion sections
            r"^(?:Future Work|Acknowledgments|Appendix|Supplementary)",  # Back matter
            r"^(?:Data Collection|Dataset|Experiments|Metrics)",  # Data/experiments
            r"^[IVXLC]+\.\s+[A-Z]",  # Roman numeral sections
        ]
        return any(re.match(pattern, line.strip()) for pattern in patterns)
