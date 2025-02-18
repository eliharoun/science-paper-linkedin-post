# science-paper-linkedin-post

## Overview
This project automates the process of generating LinkedIn posts from scientific papers on arXiv. It extracts key information from papers and generates engaging social media content using AI.

## Features
- Downloads PDF papers from arXiv links
- Extracts text content from PDFs
- Uses GPT to generate LinkedIn-style posts
- Maintains paper references

## Prerequisites
- Python 3.9+
- Required packages: `anthropic`, `requests`, `arxiv`, `nltk`, `openai`, `pymupdf`

## Setup
1. Clone the repository
2. Install Poetry (package manager):
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
3. Set up your API keys as environment variables:
    ```bash
    export ANTHROPIC_API_KEY='your-anthropic-key'
    export OPENAI_API_KEY='your-openai-key'
    ```
4. Create and activate virtual environment with dependencies:
    ```bash
    poetry install
    poetry shell
    ```

## Usage

### Process Single paper
```bash
# Process a single paper by URL
poetry run python main.py --paper-url "https://arxiv.org/abs/2312.12456" --agi-client "local"
```

### Process multiple papers
```bash
# Process multiple papers from a specific category
poetry run python main.py --category "cs.*" --max-papers 5 --agi-client "anthropic"

# Process multiple papers from a specific category and a sub-category
poetry run python main.py --category "cs.AI" --max-papers 3 --agi-client "openai"
```

### Local Model Usage with Ollama

To use locally deployed models with Ollama:

1. Install Ollama from [ollama.com/download](https://ollama.com/download)
2. Start the Ollama service
3. Pull your desired model:
    ```bash
    ollama pull <model-name>
    ```
    Example:
    ```bash
    ollama pull llama2
    ```

4. Configure your chat model with Ollama settings:
    ```python
    chat_model = ChatModelFactory.create(
         provider="ollama",
         model_name="llama3.2",
         api_base="http://localhost:11434"
    )
    ```

Available models can be found at [Ollama.com](https://ollama.com/).

#### Note
- Ensure Ollama service is running before creating the chat model
- Default port for Ollama is 11434
- Model must be pulled before it can be used
- Check Ollama documentation for specific model capabilities and requirements
