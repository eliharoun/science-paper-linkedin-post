from utils.agi_client_factory import AgiClient, LocalModelClient
import logging
from utils.paper import Paper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paper summary prompts
system_prompt = """You are an expert scientific translator who excels at distilling complex research into clear, accessible summaries for non-specialists. Your task is to create concise summaries of scientific papers that capture the core findings, significance, and implications in everyday language. Each summary must be under 300 characters, easy to understand, and provide genuine insight into the research."""


class ArxivPaperSummarizer:
    def __init__(self, agi_client: AgiClient):
        """
        Initialize the Arxiv Paper Summarizer.

        Args:
            agi_client (AgiClient): The AGI client instance used for generating content.
        """
        self.agi_client = agi_client
        self.local_mode = isinstance(self.agi_client, LocalModelClient)
        if self.local_mode:
            logger.debug("Running in local mode.")

    def summarize_paper(self, paper: Paper) -> str:
        """
        Summarize paper

        Args:
            paper: Paper object containing paper details

        Returns:
            Paper summary string
        """
        model_input = {
            "max_tokens": 2000,
            "temperature": 0,
            "system": system_prompt,
            "prompt": self._generate_user_prompt(paper),
            "pdf_path": paper.pdf_path,
        }

        try:
            return self.agi_client.get_response(model_input=model_input)

        except Exception as e:
            logger.error(f"Error generating summary for {paper['title']}: {str(e)}")
            return None

    def _generate_user_prompt(self, paper: Paper) -> str:
        return f"""Analyze this research paper title and abstract thoroughly.

       Title: {paper.title}
                        Abstract: {paper.abstract}
       
       Summarize this scientific paper and present it in an easy to understand way for a non-technical reader. 
       
       Guidelines for the summary output:
       1. Only output the summary.
       2. Avoid any additional information or interpretation.
       3. Avoid adding an introduction to the summary output, such as: 'Here is a summary of the scientific paper:', 'The paper is about', 'The provided text appears to be', or anything along those lines."""
