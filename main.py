from arxiv_paper_processor import ArxivPaperProcessor

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process arXiv papers"
    )
    parser.add_argument(
        "--paper-url", type=str, help="Single paper url to process"
    )
    parser.add_argument(
        "--max-papers", type=int, default=1, help="Maximum number of papers to process"
    )
    parser.add_argument(
        "--output", default="processed_papers.json", help="Output file path"
    )
    parser.add_argument(
        "--cache-dir", default="paper_cache", help="Cache directory for PDFs"
    )

    args = parser.parse_args()

    processor = ArxivPaperProcessor(args.cache_dir)

    if args.paper_url:
        processor.process_paper(args.paper_url)
    else:
        processor.process_papers(args.max_papers, args.output)


if __name__ == "__main__":
    main()