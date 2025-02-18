import arxiv
import requests
import json
import re
import pymupdf
import nltk
from nltk.tokenize import sent_tokenize
from paper import PaperFormat, Paper
from text_processor import TextProcessor
from typing import Dict, List, Optional
import logging
from queue import Queue
import os
from paper import Paper

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class ArxivPaperProcessor:
    def __init__(self, cache_dir: str = "paper_cache"):
        """
        Initialize the paper processor

        Args:
            cache_dir: Directory for caching downloaded PDFs
        """
        self.cache_dir = cache_dir
        self.arxiv_client = arxiv.Client()

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def get_cached_pdf_path(self, paper_id: str) -> str:
        """Get the path for a cached PDF"""
        return os.path.join(self.cache_dir, f"{paper_id}.pdf")

    def extract_pdf_content(self, pdf_url: str, paper_id: str) -> Optional[Dict]:
        """
        Extract PDF content

        Args:
            pdf_url: URL of the PDF to download
            paper_id: arXiv ID of the paper

        Returns:
            Dictionary containing extracted text and metadata
        """
        cache_path = self.get_cached_pdf_path(paper_id)

        try:
            # Check cache first
            if os.path.exists(cache_path):
                logger.info(f"Using cached PDF for {paper_id}")
                pdf_path = cache_path
            else:
                # Download PDF
                logger.info(f"Downloading PDF for {paper_id}")
                response = requests.get(pdf_url)
                response.raise_for_status()

                # Save to cache
                with open(cache_path, "wb") as f:
                    f.write(response.content)
                pdf_path = cache_path

            # Process PDF using PyMuPDF
            try:
                doc = pymupdf.open(pdf_path)
            except Exception as e:
                logger.error(f"Error opening PDF {pdf_path}: {str(e)}")
                return None

            # Extract text and maintain structure
            content = {
                "full_text": "",
                "sections": [],
                "figures": [],
                "tables": [],
                "equations": [],
                "pdf_file": pdf_path,
            }

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with formatting
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                # Check for formatting
                                is_bold = span["flags"] & 2**4 != 0
                                is_italic = span["flags"] & 2**1 != 0
                                font_size = span["size"]

                                text = span["text"]
                                # Check if the text is a title based on formatting and content
                                if is_bold and TextProcessor.is_title(text):
                                    # Add a new section with the title
                                    content["sections"].append(
                                        {
                                            "title": text,
                                            "content": "",
                                            "level": (
                                                1 if font_size > 12 else 2
                                            ),  # Determine section level based on font size
                                        }
                                    )
                                else:
                                    # Append text to the last section's content if it exists
                                    if content["sections"]:
                                        content["sections"][-1]["content"] += text + " "
                                    # Append text to the full text content
                                    content["full_text"] += text + " "
                # Extract images as metadata only
                for img_index, img in enumerate(page.get_images()):
                    content["figures"].append(
                        {"page": page_num + 1, "index": img_index, "type": "image"}
                    )

                # Convert table data to serializable format
                tables = page.find_tables()
                for table in tables:
                    serializable_table = {
                        "page": page_num + 1,
                        "rows": len(table.cells),
                        "cols": len(table.cells[0]) if table.cells else 0,
                        "type": "table",
                    }
                    content["tables"].append(serializable_table)

            # Clean the extracted text
            content["full_text"] = TextProcessor.clean_text(content["full_text"])

            # Detect paper format and structure
            paper_format = PaperFormat.detect_format(content["full_text"])
            content["format"] = paper_format

            # Close the document
            doc.close()

            return content

        except Exception as e:
            logger.error(f"Error extracting PDF content from {pdf_url}: {str(e)}")
            return None

    def fetch_recent_cs_papers(
        self, category: str = "cs.LG", max_results: int = 10
    ) -> Queue[Paper]:
        """Fetch recent CS papers from arXiv"""
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        queue = Queue()
        for result in self.arxiv_client.results(search):
            processed_paper = self.process_single_paper(result)
            if processed_paper:
                queue.put(processed_paper)

        return queue

    def fetch_paper_by_url(self, paper_url: str) -> Paper:
        """Fetch and process a single paper by its arXiv URL"""
        search = arxiv.Search(id_list=[paper_url.split("/")[-1]])

        try:
            result = next(self.arxiv_client.results(search))
            return self.process_single_paper(result)
        except Exception as e:
            logger.error(f"Error fetching paper by URL {paper_url}: {str(e)}")
            return None

    def process_single_paper(self, result: arxiv.Result) -> Paper:
        """Process a single paper"""
        try:
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                url=result.pdf_url,
                abstract=result.summary,
                arxiv_id=result.entry_id.split("/")[-1],
                primary_category=result.primary_category,
                categories=result.categories,
            )

            # Extract PDF content
            content = self.extract_pdf_content(result.pdf_url, paper.arxiv_id)
            if content:
                paper.update(content)

            return paper

        except Exception as e:
            logger.error(f"Error processing paper {result.title}: {str(e)}")
            return None

    def _extract_key_findings(self, paper: Paper) -> str:
        """Extract key findings from the paper"""
        findings = []

        # Look for results in different sections
        for section in paper.sections:
            if any(
                keyword in section["title"].lower()
                for keyword in [
                    "result",
                    "finding",
                    "evaluation",
                    "experiment",
                    "analysis",
                    "discussion",
                    "conclusion",
                    "performance",
                ]
            ):
                # Extract sentences with numerical results or key phrases
                sentences = sent_tokenize(section["content"])
                for sentence in sentences:
                    if re.search(
                        r"\d+%|\d+\.\d+|significant|improved|achieved|outperformed|decreased|increased|reduced|enhanced|superior",
                        sentence,
                    ) or any(
                        phrase in sentence.lower()
                        for phrase in [
                            "we found",
                            "shows that",
                            "demonstrates",
                            "proves",
                            "results indicate",
                            "we observed",
                            "analysis reveals",
                            "data suggests",
                            "evidence shows",
                            "study confirms",
                            "experiments demonstrate",
                            "findings suggest",
                            "we conclude",
                            "this implies",
                        ]
                    ):
                        findings.append(sentence)

        # If no structured findings, fall back to abstract analysis
        if not findings and paper.abstract:
            findings = [
                sent
                for sent in sent_tokenize(paper.abstract)
                if any(
                    phrase in sent.lower()
                    for phrase in ["we found", "shows that", "demonstrates", "proves"]
                )
            ]

        return "\n".join(findings[:3]) if findings else "No explicit findings extracted"

    def _extract_technical_innovation(self, paper: Paper) -> str:
        """Extract technical innovations from the paper"""
        innovations = []

        # Look for innovation indicators
        innovation_patterns = [
            r"novel",
            r"new",
            r"propose",
            r"introduce",
            r"develop",
            r"improve",
            r"advance",
            r"innovative",
            r"first time",
            r"outperform",
            r"better than",
            r"state-of-the-art",
            r"breakthrough",
            r"cutting-edge",
            r"pioneering",
            r"revolutionary",
            r"groundbreaking",
            r"unique",
            r"unprecedented",
            r"enhance",
            r"optimize",
            r"refine",
            r"transform",
            r"novel approach",
            r"efficient",
            r"robust",
            r"scalable",
            r"versatile",
            r"effective",
            r"superior",
            r"high-performance",
            r"next-generation",
            r"leading",
            r"trailblazing",
            r"innovate",
            r"benchmark",
            r"exceed",
            r"surpass",
            r"redefine",
            r"paradigm shift",
        ]

        # Check methodology and approach sections
        for section in paper.sections:
            if any(
                keyword in section["title"].lower()
                for keyword in [
                    "method",
                    "approach",
                    "technique",
                    "methodology",
                    "design",
                    "procedure",
                    "mechanism",
                    "strategy",
                    "process",
                    "workflow",
                ]
            ):
                sentences = sent_tokenize(section["content"])
                for sentence in sentences:
                    if any(
                        re.search(pattern, sentence.lower())
                        for pattern in innovation_patterns
                    ):
                        innovations.append(sentence)

        # Fall back to abstract if needed
        if not innovations and paper.abstract:
            sentences = sent_tokenize(paper.abstract)
            innovations = [
                sent
                for sent in sentences
                if any(
                    re.search(pattern, sent.lower()) for pattern in innovation_patterns
                )
            ]

        return (
            "\n".join(innovations[:2])
            if innovations
            else "No explicit innovations extracted"
        )

    def _extract_practical_applications(self, paper: Paper) -> str:
        """Extract practical applications from the paper"""
        applications = []

        # Application-related keywords
        application_patterns = [
            r"application",
            r"use case",
            r"practical",
            r"industry",
            r"real-world",
            r"implement",
            r"deploy",
            r"utilize",
            r"apply",
            r"adapt",
            r"integrate",
            r"field",
            r"commercial",
            r"market",
            r"solution",
            r"prototype",
            r"product",
            r"service",
            r"tool",
            r"framework",
        ]

        # Check discussion and conclusion sections
        for section in paper.sections:
            if any(
                keyword in section["title"].lower()
                for keyword in ["discussion", "conclusion", "application"]
            ):
                sentences = sent_tokenize(section["content"])
                for sentence in sentences:
                    if any(
                        re.search(pattern, sentence.lower())
                        for pattern in application_patterns
                    ):
                        applications.append(sentence)

        return (
            "\n".join(applications[:2])
            if applications
            else "No explicit applications extracted"
        )

    def _extract_impact(self, paper: Paper) -> str:
        """Extract potential impact from the paper"""
        impact_statements = []

        # Impact-related keywords
        impact_patterns = [
            r"impact",
            r"benefit",
            r"potential",
            r"future work",
            r"promise",
            r"implications",
            r"contribute",
            r"significance",
            r"importance",
            r"effect",
            r"influence",
            r"advancement",
            r"progress",
            r"outcome",
            r"result",
            r"consequence",
            r"repercussion",
            r"ramification",
            r"prospect",
            r"advantage",
        ]

        # Check conclusion and discussion sections
        for section in paper.sections:
            if any(
                keyword in section["title"].lower()
                for keyword in ["conclusion", "discussion"]
            ):
                sentences = sent_tokenize(section["content"])
                for sentence in sentences:
                    if any(
                        re.search(pattern, sentence.lower())
                        for pattern in impact_patterns
                    ):
                        impact_statements.append(sentence)

        return (
            "\n".join(impact_statements[:2])
            if impact_statements
            else "No explicit impact statements extracted"
        )

    def process_papers(
        self,
        category: str = "cs.LG",
        max_results: int = 10,
        output_file: str = "processed_papers.json",
    ) -> List[Paper]:
        """
        Main function to process papers

        Args:
            output_file: Path to save the processed papers
        """
        try:
            logger.info("Fetching recent CS papers...")
            papers = self.fetch_recent_cs_papers(category, max_results)

            # Process each paper
            processed_papers = []
            while not papers.empty():
                try:
                    paper = papers.get()
                    logger.info(f"Processing paper: {paper.arxiv_id}")

                    paper.key_findings = self._extract_key_findings(paper)
                    paper.technical_innovation = self._extract_technical_innovation(
                        paper
                    )
                    paper.practical_applications = self._extract_practical_applications(
                        paper
                    )
                    paper.impact_analysis = self._extract_impact(paper)
                    processed_papers.append(paper)
                except Exception as e:
                    logger.error(f"Error processing individual paper: {str(e)}")
                    continue

            # Save results
            serialized_papers = [paper.to_dict() for paper in processed_papers]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serialized_papers, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Processed {len(processed_papers)} papers successfully and saved to {output_file}"
            )

            return processed_papers

        except Exception as e:
            logger.error(f"Error in fetching and processing papers: {str(e)}")
            raise

    def process_paper(
        self, paper_url: str, output_file: str = "processed_papers.json"
    ) -> Paper:
        """
        Main function to process paper by url

        Args:
            paper_url: Url to arxiv paper
            output_file: Path to save the processed papers
        """
        try:
            logger.info(f"Fetching paper: {paper_url}")
            paper = self.fetch_paper_by_url(paper_url=paper_url)

            logger.info(f"Processing paper: {paper.arxiv_id}")
            paper.key_findings = self._extract_key_findings(paper)
            paper.technical_innovation = self._extract_technical_innovation(paper)
            paper.practical_applications = self._extract_practical_applications(paper)
            paper.impact_analysis = self._extract_impact(paper)
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            raise

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Processed paper and saved to {output_file}")

        return paper
