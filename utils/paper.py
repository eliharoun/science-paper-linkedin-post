from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import re


@dataclass
class Paper:
    title: str
    authors: List[str]
    published: str
    url: str
    abstract: str
    arxiv_id: str
    primary_category: str
    categories: List[str]
    format: str = "standard"
    full_text: Optional[str] = None
    sections: Optional[List[Dict]] = None
    figures: Optional[List[Dict]] = None
    tables: Optional[List[Dict]] = None
    equations: Optional[List[Dict]] = None
    key_findings: Optional[str] = None
    technical_innovation: Optional[str] = None
    practical_applications: Optional[str] = None
    impact_analysis: Optional[str] = None
    pdf_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert the Paper object to a dictionary"""
        return {
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "url": self.url,
            "abstract": self.abstract,
            "arxiv_id": self.arxiv_id,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "format": self.format,
            "key_findings": self.key_findings,
            "technical_innovation": self.technical_innovation,
            "practical_applications": self.practical_applications,
            "impact_analysis": self.impact_analysis,
            "pdf_path": self.pdf_path,
        }

    def to_dict_full(self) -> Dict:
        """Convert the complete Paper object to a dictionary"""
        return {
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "url": self.url,
            "abstract": self.abstract,
            "arxiv_id": self.arxiv_id,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "format": self.format,
            "full_text": self.full_text,
            "sections": self.sections,
            "figures": self.figures,
            "tables": self.tables,
            "equations": self.equations,
            "key_findings": self.key_findings,
            "technical_innovation": self.technical_innovation,
            "practical_applications": self.practical_applications,
            "impact_analysis": self.impact_analysis,
            "pdf_path": self.pdf_path,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Paper":
        """Create a Paper object from a dictionary"""
        return cls(**data)

    def update(self, content: dict):
        """Update paper details with additional content"""
        for key, value in content.items():
            if hasattr(self, key):
                setattr(self, key, value)


class PaperFormat:
    """Class to handle different research paper formats"""

    # Common section patterns for different paper formats
    FORMATS = {
        "standard": {
            "abstract": r"^abstract",
            "introduction": r"^(?:1\.\s*)?introduction",
            "related_work": r"^(?:\d\.\s*)?related work",
            "background": r"^(?:\d\.\s*)?background",
            "methodology": r"^(?:2\.\s*)?(?:methodology|methods|experimental setup)",
            "implementation": r"^(?:\d\.\s*)?implementation",
            "results": r"^(?:3\.\s*)?(?:results|findings|evaluation)",
            "discussion": r"^(?:4\.\s*)?discussion",
            "future_work": r"^(?:\d\.\s*)?future work",
            "conclusion": r"^(?:5\.\s*)?(?:conclusion|summary)",
            "acknowledgments": r"^acknowledgments?",
            "references": r"^(?:6\.\s*)?(?:references|bibliography)",
            "appendix": r"^appendix",
        },
        "ieee": {
            "abstract": r"^abstract",
            "keywords": r"^(?:index terms|keywords)",
            "introduction": r"^i\.\s*introduction",
            "related_work": r"^ii\.\s*(?:related work|background)",
            "methodology": r"^(?:ii|iii)\.\s*(?:methodology|proposed method)",
            "system_model": r"^(?:ii|iii)\.\s*system model",
            "results": r"^(?:iii|iv)\.\s*(?:results|experimental results)",
            "analysis": r"^(?:iv|v)\.\s*analysis",
            "conclusion": r"^(?:v|vi)\.\s*conclusion",
            "acknowledgment": r"^acknowledgment",
            "references": r"^references",
        },
        "acm": {
            "abstract": r"^abstract",
            "keywords": r"^(?:keywords|general terms)",
            "introduction": r"^1\.\s*introduction",
            "background": r"^2\.\s*(?:background|related work)",
            "methodology": r"^3\.\s*(?:methodology|approach)",
            "design": r"^(?:\d\.\s*)?(?:design|architecture)",
            "evaluation": r"^4\.\s*evaluation",
            "discussion": r"^(?:\d\.\s*)?discussion",
            "conclusion": r"^5\.\s*conclusion",
            "acknowledgments": r"^acknowledgments",
            "references": r"^references",
        },
    }

    @classmethod
    def detect_format(cls, text: str) -> str:
        """Detect the paper format based on section patterns"""
        format_scores = defaultdict(int)

        for format_name, patterns in cls.FORMATS.items():
            for section, pattern in patterns.items():
                if re.search(pattern, text.lower()):
                    format_scores[format_name] += 1

        return (
            max(format_scores.items(), key=lambda x: x[1])[0]
            if format_scores
            else "standard"
        )


class PaperSection:
    """Class to represent a section of a research paper"""

    def __init__(self, title: str, content: str, level: int = 1):
        self.title = title
        self.content = content
        self.level = level
        self.subsections: List[PaperSection] = []

    def add_subsection(self, subsection: "PaperSection"):
        self.subsections.append(subsection)

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [s.to_dict() for s in self.subsections],
        }
