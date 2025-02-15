import argparse
import logging

from arxiv_paper_processor import ArxivPaperProcessor


def main():
    """Main entry point"""

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process arXiv papers")

    # Add argument for single paper URL
    parser.add_argument("--paper-url", type=str, help="Single paper url to process")

    # Add argument for category of papers to process (default is Machine Learning "cs.LG"). Ref: https://arxiv.org/category_taxonomy
    # Used if paper URL is not provide
    parser.add_argument(
        "--category",
        default="cs.LG",
        help="Category of papers to process (default is 'cs.AI')",
    )

    # Add argument for maximum number of papers to process (default is 1). Used if paper URL is not provide
    parser.add_argument(
        "--max-papers", type=int, default=1, help="Maximum number of papers to process"
    )

    # Add argument for output file path. Default is "processed_papers.json"
    parser.add_argument(
        "--output", default="processed_papers.json", help="Output file path"
    )

    # Add argument for cache directory for PDFs published on arXiv. Default is "paper_cache"
    parser.add_argument(
        "--cache-dir",
        default="paper_cache",
        help="Cache directory for PDFs published on arXiv",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create an instance of ArxivPaperProcessor with the specified cache directory
    processor = ArxivPaperProcessor(args.cache_dir)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process a single paper if paper URL is provided
    if args.paper_url:
        logging.info(f"Processing single paper from URL: {args.paper_url}")
        processor.process_paper(args.paper_url)

        logging.info("Finished processing paper")
    else:
        # Otherwise, process multiple papers up to the specified maximum number
        logging.info(
            f"Processing the most recent {args.max_papers} papers, saving to {args.output}"
        )
        processor.process_papers(args.category, args.max_papers, args.output)

        logging.info("Finished processing papers")


if __name__ == "__main__":
    # Entry point for the script
    main()
