import argparse
import logging
import time

from utils.arxiv_paper_processor import ArxivPaperProcessor
from utils.agi_client_factory import AgiClientFactory
from utils.linkedin_post_generator import LinkedInPostGenerator
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

    # Add argument for processed paper output file path. Default is "output/processed_papers.json"
    parser.add_argument(
        "--processed-paper-output",
        default="output/processed_papers.json",
        help="Processed paper output file path",
    )

    # Add argument for cache directory for PDFs published on arXiv. Default is "output/paper_cache"
    parser.add_argument(
        "--cache-dir",
        default="output/paper_cache",
        help="Cache directory for PDFs published on arXiv",
    )

    # Add argument for output file path for the LinkedIn post content. Default is "output/linkedin_posts.json"
    parser.add_argument(
        "--output",
        default="output/linkedin_posts.json",
        help="Output file path for the LinkedIn post content",
    )

    parser.add_argument(
        "--agi-client",
        default="local",
        help="AGI client to use for generating LinkedIn post content. Choose from 'local', 'openai' or 'anthropic'. Default is 'local'",
    )

    parser.add_argument(
        "--model-name",
        help="Model to use for generating LinkedIn post content. Model must be supported by the AGI client.",
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

    # Check if model name is provided for the AGI client. If not, set the default model name based on the AGI client
    if args.agi_client == "local" and not args.model_name:
        args.model_name = "llama3.2"
    elif args.agi_client == "openai" and not args.model_name:
        args.model_name = "gpt-3.5-turbo"
    elif args.agi_client == "anthropic" and not args.model_name:
        args.model_name = "claude-3-5-sonnet-20241022"

    # Create an instance of the AGI client
    agi_client = AgiClientFactory.create_client(args.agi_client, args.model_name)

    # Create an instance of the LinkedInPostGenerator
    linkedin_post_generator = LinkedInPostGenerator(agi_client)

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
                    "url": paper.url,
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
            output_list = []
            for paper in papers:
                # Generate a LinkedIn post for each processed paper
                post_content = linkedin_post_generator.generate_linkedin_post(paper)

                output = {
                    "title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "published": paper.published,
                    "authors": paper.authors,
                    "url": paper.url,
                    "linkedin_post": post_content,
                }

                output_list.append(output)

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_list, f, ensure_ascii=False, indent=2)

            logging.info(f"LinkedIn posts generated successfully")


if __name__ == "__main__":
    # Entry point for the script
    main()
