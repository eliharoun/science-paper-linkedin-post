import arxiv
import requests
import json
import re
import pymupdf
import nltk
from nltk.tokenize import sent_tokenize
from utils.paper import PaperFormat, Paper
from utils.text_processor import TextProcessor
from typing import Dict, List, Optional
import logging
import os
from utils.paper import Paper
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class ArxivPaperProcessor:
    def __init__(self, cache_dir: str = "output/paper_cache"):
        """
        Initialize the paper processor

        Args:
            cache_dir: Directory for caching downloaded PDFs
        """
        self.cache_dir = cache_dir
        self.arxiv_client = arxiv.Client()

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_process_recent_papers(
        self, category: str = "cs.LG", max_results: int = 10, process_pdf=False
    ) -> List[Paper]:
        """
        Main function to fetch and process recent papers

        Args:
            category: ArXiv category to search
            max_results: Maximum number of papers to fetch
        """
        try:
            logger.info("Fetching recent papers...")
            arxiv_results = self.fetch_recent_from_arxiv(category, max_results)

            # Only process PDF if explicitly requested
            if not process_pdf:
                # Just fetch metadata
                papers: list[Paper] = [
                    self.create_paper(result) for result in arxiv_results
                ]
            else:
                papers: list[Paper] = self.process_arxiv_results(arxiv_results)

            return papers
        except Exception as e:
            logger.error(f"Error in fetching and processing papers: {str(e)}")
            raise

    def fetch_process_single_paper_url(
        self, paper_url: str, process_pdf=False
    ) -> Paper:
        """
        Main function to process paper by url

        Args:
            paper_url: Url to arxiv paper
        """
        logger.info(f"Fetching paper: {paper_url}")
        try:
            # Only process PDF if explicitly requested
            if not process_pdf:
                # Just fetch metadata
                result = self.fetch_from_arxiv_by_url(paper_url)[0]
                paper = self.create_paper(result)
            else:
                arxiv_results = self.fetch_from_arxiv_by_url(paper_url)
                paper = self.process_arxiv_results(arxiv_results)[0]
        except Exception as e:
            logger.error(f"Error in fetching and processing paper: {str(e)}")
            raise

        return paper

    def fetch_process_single_paper_id(self, paper_id: str, process_pdf=False) -> Paper:
        """
        Main function to process paper by id

        Args:
            paper_id: Id of arxiv paper
        """
        try:
            # Only process PDF if explicitly requested
            if not process_pdf:
                # Just fetch metadata
                arxiv_results = self.fetch_from_arxiv_by_id(paper_id)[0]
                paper = self.create_paper(arxiv_results)
            else:
                arxiv_results = self.fetch_from_arxiv_by_id(paper_id)
                paper = self.process_arxiv_results(arxiv_results)[0]
        except Exception as e:
            logger.error(f"Error in fetching and processing paper: {str(e)}")
            raise

        return paper

    def fetch_recent_from_arxiv(
        self, category: str = "cs.LG", max_results: int = 10
    ) -> list[arxiv.Result]:
        """Fetch recent papers from arXiv

        Args:
            category: ArXiv category to search
            max_results: Maximum number of papers to fetch
        """
        try:
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            return list(self.arxiv_client.results(search))
        except Exception as e:
            logger.error(f"Error fetching papers from arXiv: {str(e)}")
            raise

    def fetch_from_arxiv_by_id(self, paper_id: str) -> list[arxiv.Result]:
        """Fetch paper from arXiv by paper id"""
        try:
            search = arxiv.Search(id_list=[paper_id])
            return list(self.arxiv_client.results(search))
        except Exception as e:
            logger.error(f"Error fetching papers from arXiv: {str(e)}")
            raise

    def fetch_from_arxiv_by_url(self, paper_url: str) -> list[arxiv.Result]:
        """Fetch paper from arXiv for a given url"""
        paper_id = self.extract_arxiv_id_from_url(paper_url)
        return self.fetch_from_arxiv_by_id(paper_id)

    def process_arxiv_results(self, arxiv_results: list[arxiv.Result]) -> list[Paper]:
        """
        Process arXiv results by loading cached papers or processing new ones.

        Args:
            arxiv_results: List of arXiv search results

        Returns:
            List of processed Paper objects
        """
        processed_papers = self._load_cached_papers(arxiv_results)
        new_papers = self._get_unprocessed_papers(arxiv_results)

        if new_papers:
            new_papers = self._download_pdfs_parallel(new_papers)
            processed_new_papers = self._process_new_papers(new_papers)
            processed_papers.extend(processed_new_papers)

        return processed_papers

    def extract_arxiv_id_from_url(self, url: str) -> str:
        return url.split("/")[-1]

    def create_paper(self, result: arxiv.Result) -> Paper:
        """Create a Paper object from an arXiv result"""
        return Paper(
            title=result.title,
            authors=[author.name for author in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
            url=result.pdf_url,
            abstract=result.summary,
            arxiv_id=self.extract_arxiv_id_from_url(result.entry_id),
            primary_category=result.primary_category,
            categories=result.categories,
        )

    def _load_cached_papers(self, arxiv_results: list[arxiv.Result]) -> list[Paper]:
        """Load already processed papers from cache."""
        cached_papers = []
        for result in arxiv_results:
            paper_id = self.extract_arxiv_id_from_url(result.entry_id)
            if self._processed_paper_exists(paper_id):
                logger.info(f"Loading cached processed paper for {paper_id}")
                cached_papers.append(self._get_processed_paper(paper_id))
        return cached_papers

    def _get_unprocessed_papers(self, arxiv_results: list[arxiv.Result]) -> list[Paper]:
        """Create Paper objects for results not in cache."""
        return [
            self.create_paper(result)
            for result in arxiv_results
            if not self._processed_paper_exists(
                self.extract_arxiv_id_from_url(result.entry_id)
            )
        ]

    def _download_pdfs_parallel(self, papers: list[Paper]) -> list[Paper]:
        """Download PDFs in parallel for all papers."""
        max_workers = min(len(papers), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {
                executor.submit(self._download_pdf, paper.url, paper.arxiv_id): paper
                for paper in papers
            }

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                paper.pdf_path = future.result()

        return papers

    def _process_new_papers(self, papers: list[Paper]) -> list[Paper]:
        """Process and cache new papers."""
        processed_papers = []
        for paper in papers:
            processed_paper = self._process_single_paper(paper)
            if processed_paper:
                self._save_processed_paper(processed_paper)
                processed_papers.append(processed_paper)
        return processed_papers

    def _get_cached_pdf_path(self, paper_id: str) -> str:
        """Get the path for a cached PDF"""
        cached_pdf_dir = os.path.join(self.cache_dir, "pdf_files")
        os.makedirs(cached_pdf_dir, exist_ok=True)
        return os.path.join(cached_pdf_dir, f"{paper_id}.pdf")

    def _download_pdf(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """
        Download a PDF file
        """
        cache_path = self._get_cached_pdf_path(paper_id)

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

            return pdf_path
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
            return None

    def _extract_pdf_content(self, pdf_path: str) -> Optional[Dict]:
        """
        Extract PDF content

        Args:
            pdf_url: URL of the PDF to download
            paper_id: arXiv ID of the paper

        Returns:
            Dictionary containing extracted text and metadata
        """
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
            "pdf_path": pdf_path,
        }

        in_references = False
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
                            # Check if we're entering references section
                            if (
                                is_bold
                                and TextProcessor.is_title(text)
                                and text.lower().strip() == "references"
                            ):
                                in_references = True
                                continue

                            # Skip if we're in references section
                            if in_references:
                                continue

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

    def _process_single_paper(self, paper: Paper) -> Paper:
        """Extract content from a single paper and update the Paper object"""
        try:
            # Extract PDF content
            content = self._extract_pdf_content(paper.pdf_path)
            if content:
                paper.update(content)

            paper.key_findings = self._extract_key_findings(paper)
            paper.technical_innovation = self._extract_technical_innovation(paper)
            paper.practical_applications = self._extract_practical_applications(paper)
            paper.impact_analysis = self._extract_impact(paper)

            return paper

        except Exception as e:
            logger.error(f"Error processing paper {paper.title}: {str(e)}")
            return None

    def _create_get_cached_processed_paper_dir(self) -> str:
        """Create and return the directory for cached processed papers"""
        processed_papers_dir = os.path.join(self.cache_dir, "processed_papers")
        os.makedirs(processed_papers_dir, exist_ok=True)
        return processed_papers_dir

    def _save_processed_paper(self, paper: Paper):
        """Save processed paper content to JSON file"""
        try:
            # Create the processed papers directory if it doesn't exist
            cached_processed_paper_path = os.path.join(
                self._create_get_cached_processed_paper_dir(), f"{paper.arxiv_id}.json"
            )

            with open(cached_processed_paper_path, "w", encoding="utf-8") as f:
                json.dump(paper.to_dict_full(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Processed paper saved to {cached_processed_paper_path}")
        except IOError as e:
            logger.error(f"Failed to save processed paper {paper.arxiv_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error saving paper {paper.arxiv_id}: {str(e)}")

    def _processed_paper_exists(self, paper_id: str) -> bool:
        """Check if a processed paper exists in the cache"""
        cached_processed_paper_path = os.path.join(
            self._create_get_cached_processed_paper_dir(), f"{paper_id}.json"
        )
        logger.debug(
            f"Checking if processed paper exists for paper: {paper_id} at path : {cached_processed_paper_path}"
        )
        return os.path.exists(cached_processed_paper_path)

    def _get_processed_paper(self, paper_id: str) -> Paper:
        """Get a processed paper from the cache"""
        cached_processed_paper_path = os.path.join(
            self._create_get_cached_processed_paper_dir(), f"{paper_id}.json"
        )
        with open(cached_processed_paper_path, "r", encoding="utf-8") as f:
            return Paper.from_dict(json.load(f))

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
