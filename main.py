from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  # Add this import

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
import uvicorn
from cachetools import LRUCache
from utils.arxiv_paper_processor import ArxivPaperProcessor
from utils.agi_client_factory import AgiClientFactory
from utils.linkedin_post_generator import LinkedInPostGenerator
from utils.arxiv_paper_summarizer import ArxivPaperSummarizer
import asyncio
import logging
from utils.paper import Paper
import time
import psutil

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

start_time = time.time()

app = FastAPI(
    title="Research Papers API",
    description="API for serving research paper data",
    version="1.0",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

paper_processor = ArxivPaperProcessor()
agi_client_factory = AgiClientFactory()

local_agi_client = agi_client_factory.create_client("local", "llama3.2")
anthropic_client = agi_client_factory.create_client(
    "anthropic", "claude-3-5-sonnet-20241022"
)

post_generator = LinkedInPostGenerator(anthropic_client)
paper_summarizer = ArxivPaperSummarizer(local_agi_client)

# In-memory cache with LRU eviction policy
papers_cache = LRUCache(maxsize=150)


# Paper data model
class PaperSummary(BaseModel):
    paper_id: str
    title: str
    published: str
    authors: List[str]
    url: HttpUrl
    paper_summary: str
    categories: List[str]


# LinkedIn post data model
class LinkedInPost(BaseModel):
    paper_id: str
    title: str
    published: str
    authors: List[str]
    url: HttpUrl
    linkedin_post: str


def update_paper_cache(paper_data):
    """Update papers_cache cache"""
    papers_cache[paper_data["paper_id"]] = paper_data


def save_to_disk(text: str, output_file_path: str):
    """Write the output to the output file"""
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


async def fetch_paper_by_url(paper_url: str) -> Dict[str, Any]:
    """Fetch paper details by URL"""

    paper: Paper = paper_processor.fetch_process_single_paper_url(paper_url)
    paper_summary = paper_summarizer.summarize_paper(paper)

    return {
        "paper_id": paper.arxiv_id,
        "title": paper.title,
        "published": paper.published,
        "authors": paper.authors,
        "url": paper.url,
        "paper_summary": paper_summary,
        "categories": paper.categories,
    }


async def fetch_paper_by_id(paper_id: str) -> Dict[str, Any]:
    paper = paper_processor.fetch_process_single_paper_id(paper_id)
    paper_summary = paper_summarizer.summarize_paper(paper)

    return {
        "paper_id": paper.arxiv_id,
        "title": paper.title,
        "published": paper.published,
        "authors": paper.authors,
        "url": paper.url,
        "paper_summary": paper_summary,
        "categories": paper.categories,
    }


async def summarize_paper(paper: Paper) -> str:
    return paper_summarizer.summarize_paper(paper)


async def fetch_recent_papers(results: list[Any]) -> List[Dict[str, Any]]:
    papers: list[Paper] = [paper_processor.create_paper(result) for result in results]

    papers_summaries = {}
    try:
        tasks = {
            paper.arxiv_id: asyncio.create_task(summarize_paper(paper))
            for paper in papers
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Get results
        for paper_id, task in tasks.items():
            papers_summaries[paper_id] = task.result()
    except Exception as e:
        logger.error(f"Error summarizing paper: {e}")

    summarized_papers = []
    for paper in papers:
        summarized_papers.append(
            {
                "paper_id": paper.arxiv_id,
                "title": paper.title,
                "published": paper.published,
                "authors": paper.authors,
                "url": paper.url,
                "paper_summary": papers_summaries[paper.arxiv_id],
                "categories": paper.categories,
            }
        )

    return summarized_papers


async def fetch_linkedin_post(paper_id: str) -> Dict[str, Any]:
    paper: Paper = paper_processor.fetch_process_single_paper_id(paper_id, True)
    linkedin_post = post_generator.generate_linkedin_post(paper)

    return {
        "paper_id": paper_id,
        "title": paper.title,
        "published": paper.published,
        "authors": paper.authors,
        "url": paper.url,
        "linkedin_post": linkedin_post,
    }


# API Endpoints
@app.get("/api/paper/url", response_model=PaperSummary)
async def get_single_paper_by_url(
    paper_url: HttpUrl = Query(..., description="URL of the research paper"),
    background_tasks: BackgroundTasks = None,
):
    """Get paper details by URL"""
    # Check cache first
    paper_id = paper_processor.extract_arxiv_id_from_url(paper_url)
    if paper_id in papers_cache:
        logger.info("Returning cached result")
        return papers_cache[paper_id]

    # If not in cache, fetch from database
    try:
        paper_data = await asyncio.wait_for(
            fetch_paper_by_url(str(paper_url)), timeout=10.0
        )

        if background_tasks:
            background_tasks.add_task(update_paper_cache, paper_data)

        return paper_data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Paper not found: {str(e)}")


@app.get("/api/paper/id/{paper_id}", response_model=PaperSummary)
async def get_single_paper_by_id(
    paper_id: str, background_tasks: BackgroundTasks = None
):
    """Get paper details by ID"""
    # Check cache first
    if paper_id in papers_cache:
        logger.info("Returning cached result")
        return papers_cache[paper_id]

    # If not in cache, fetch from database
    try:
        paper_data = await asyncio.wait_for(
            fetch_paper_by_id(str(paper_id)), timeout=10.0
        )

        if background_tasks:
            background_tasks.add_task(update_paper_cache, paper_data)

        return paper_data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Paper not found: {str(e)}")


@app.get("/api/papers/recent", response_model=List[PaperSummary])
async def get_recent_papers(
    max_num_papers: int = Query(
        5, description="Maximum number of papers to return", ge=1, le=100
    ),
    category: str = Query("cs.LG", description="Research category"),
    background_tasks: BackgroundTasks = None,
):
    """Get a list of recent papers in a category"""
    try:
        # Fetch recent from arXiv
        arxiv_results = paper_processor.fetch_recent_from_arxiv(
            category, max_num_papers
        )
        recent_paper_ids = set(
            [
                paper_processor.extract_arxiv_id_from_url(result.entry_id)
                for result in arxiv_results
            ]
        )

        cached_paper_ids = set(papers_cache.keys())
        ids_to_process = recent_paper_ids - cached_paper_ids
        results_to_process = [
            result
            for result in arxiv_results
            if paper_processor.extract_arxiv_id_from_url(result.entry_id)
            in ids_to_process
        ]

        # Process papers that aren't cached
        new_papers = await fetch_recent_papers(results_to_process)

        response_papers = list(
            paper
            for paper in papers_cache.values()
            if paper["paper_id"] in recent_paper_ids
        )

        for paper in new_papers:
            response_papers.append(paper)
            if background_tasks:
                background_tasks.add_task(update_paper_cache, paper)

        return response_papers
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch recent papers: {str(e)}"
        )


@app.get("/api/linkedin/post/{paper_id}", response_model=LinkedInPost)
async def get_linkedin_post_by_id(
    paper_id: str,
    background_tasks: BackgroundTasks = None,
):
    """Get LinkedIn post content for a paper"""
    try:
        linkedin_data = await fetch_linkedin_post(paper_id)

        if background_tasks:
            background_tasks.add_task(
                save_to_disk, linkedin_data["linkedin_post"], "linkedin_post.txt"
            )

        return linkedin_data
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"LinkedIn post not found: {str(e)}"
        )


@app.get("/api/linkedin/post/url", response_model=LinkedInPost)
async def get_linkedin_post_by_url(
    paper_url: HttpUrl = Query(..., description="URL of the research paper"),
    background_tasks: BackgroundTasks = None,
):
    """Get LinkedIn post content for a paper"""
    try:
        paper_id = paper_processor.extract_arxiv_id_from_url(paper_url)
        linkedin_data = await fetch_linkedin_post(paper_id)

        if background_tasks:
            background_tasks.add_task(
                save_to_disk, linkedin_data["linkedin_post"], "linkedin_post.txt"
            )

        return linkedin_data
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"LinkedIn post not found: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_size": len(papers_cache),
        "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "uptime_seconds": time.time() - start_time,
    }


# Clear cache endpoint (for maintenance)
@app.post("/api/admin/clear-cache", status_code=204)
async def clear_cache():
    papers_cache.clear()
    return None


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
