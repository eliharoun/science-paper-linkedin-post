import argparse
import logging

from arxiv_paper_processor import ArxivPaperProcessor
from agi_client_factory import AgiClientFactory
from linkedin_post_generator import LinkedInPostGenerator
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process arXiv papers")

    # Add argument for single paper URL
    parser.add_argument("--paper-url", type=str, help="Single paper url to process")

    # Add argument for category of papers to process (default is Machine Learning "cs.LG"). Ref: https://arxiv.org/category_taxonomy
    # Used if paper URL is not provide
    parser.add_argument(
        "--category",
        default="cs.LG",
        help="Category of papers to process (default is 'cs.AI'). Used if --paper-url is not provide",
    )

    # Add argument for maximum number of papers to process (default is 1). Used if --paper-url is not provide
    parser.add_argument(
        "--max-papers",
        type=int,
        default=1,
        help="Maximum number of papers to process. Used if --paper-url is not provide",
    )

    # Add argument for processed paper output file path. Default is "processed_papers.json"
    parser.add_argument(
        "--processed-paper-output",
        default="processed_papers.json",
        help="Processed paper output file path",
    )

    # Add argument for cache directory for PDFs published on arXiv. Default is "paper_cache"
    parser.add_argument(
        "--cache-dir",
        default="paper_cache",
        help="Cache directory for PDFs published on arXiv",
    )

    # Add argument for output file path for the LinkedIn post content. Default is "linkedin_posts.json"
    parser.add_argument(
        "--output",
        default="linkedin_posts.json",
        help="Output file path for the LinkedIn post content",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    # Get command line arguments
    args = parse_arguments()

    logging.info(
        f"Starting arXiv paper processing with the following arguments: {args}"
    )

    # Create an instance of ArxivPaperProcessor with the specified cache directory
    processor = ArxivPaperProcessor(args.cache_dir)

    # Create an instance of the AGI client
    agi_client = AgiClientFactory.create_client("anthropic")

    # Create an instance of the LinkedInPostGenerator
    linkedin_post_generator = LinkedInPostGenerator(agi_client, model_name="claude-3-5-sonnet-20241022")

    # Process a single paper if paper URL is provided
    if args.paper_url:
        logging.info(f"Processing single paper from URL: {args.paper_url}")
        paper = processor.process_paper(args.paper_url)

        if paper:
            # Generate a LinkedIn post for the processed paper
            post_content = linkedin_post_generator.generate_linkedin_post(paper)

            if post_content:
                logging.info(f"LinkedIn post generated successfully")
                output = {
                    "title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "published": paper.published,
                    "authors": paper.authors,
                    "linkedin_post": post_content,
                }
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

                logging.info(f"Output saved to {args.output}")
            else:
                logging.error("Error generating LinkedIn post")

    else:
        # Otherwise, process multiple papers up to the specified maximum number
        logging.info(
            f"Processing the most recent {args.max_papers} papers, saving to {args.output}"
        )
        papers = processor.process_papers(
            args.category, args.max_papers, args.processed_paper_output
        )

        if papers:
            posts_content = []
            output_list = []
            for paper in papers:
                # Generate a LinkedIn post for each processed paper
                post_content = linkedin_post_generator.generate_linkedin_post(paper)
                posts_content.append(post_content)

                output = {
                    "title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "published": paper.published,
                    "authors": paper.authors,
                    "linkedin_post": post_content,
                }

                output_list.append(output)

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_list, f, ensure_ascii=False, indent=2)

            logging.info(f"LinkedIn posts generated successfully")


if __name__ == "__main__":
    # Entry point for the script
    main()
