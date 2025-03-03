# ArXiv Science Paper LinkedIn Post Generator

## Overview
This project automates the process of generating LinkedIn posts from scientific papers on arXiv. It extracts key information from papers and generates engaging social media content using AI.

## Features
- Downloads PDF papers from arXiv links
- Extracts text content from PDFs
- Uses GPT to generate LinkedIn-style posts
- Maintains paper references

### Supported AI Models

#### Cloud-based Models
- **OpenAI API**
  - Supports GPT-3.5-turbo and GPT-4 models and others
  - Requires API key configuration

- **Claude.ai (Anthropic)**
  - Supports Claude 3.5 Sonnet, Claude 3.5 Haiku and others
  - Requires Anthropic API key
  - Supports uploading and analyzing entire PDF file

#### Local Models
- **Ollama Integration**
  - Run AI models locally without cloud dependencies
  - Supports various open-source models like:
    - Deepseek R1
    - Llama 3.2
    - Mistral
  - No API keys required
  - Suitable for privacy-focused deployments

## Prerequisites
- Python 3.11+
- Required packages: `anthropic`, `requests`, `arxiv`, `nltk`, `openai`, `pymupdf`, `ollama`, `tenacity`, `fastapi`, `uvicorn`, `cachetools`

## Get Everything Ready

### Local Model Usage with Ollama (Optional)

To use locally deployed models with Ollama:

1. Install Ollama from [ollama.com/download](https://ollama.com/download)
2. Start the Ollama service
3. Pull your desired model:
    ```bash
    ollama pull <model-name>
    ```
    Example:
    ```bash
    ollama pull llama3.2
    ```

4. Run model:
    ```bash
    ollama run llama3.2
    ```

Available models can be found at [Ollama.com](https://ollama.com/).

#### Note
- Ensure Ollama service is running before creating the chat model
- Default port for Ollama is 11434
- Model must be pulled before it can be used
- Check Ollama documentation for specific model capabilities and requirements

### Required Setup
1. Clone the repository
2. Install Poetry (package manager):
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
3. Set up your API keys as environment variables if using Anthropic or OpenAI:
    ```bash
    export ANTHROPIC_API_KEY='your-anthropic-key'
    export OPENAI_API_KEY='your-openai-key'
    ```
4. Create and activate virtual environment with dependencies:
    ```bash
    poetry install
    poetry shell
    ```

## CLI Tool Usage

### Supported Input Arguments

- `--paper-url`: Specifies a single paper URL to process
- `--category`: Category of papers to process (default: 'cs.LG'). Only used when paper URL is not provided 
- `--max-papers`: Maximum number of papers to process (default: 1). Only used when paper URL is not provided
- `--processed-paper-output`: File path for processed paper output (default: "processed_papers.json")
- `--cache-dir`: Directory path for caching arXiv PDFs (default: "paper_cache")
- `--output`: Output file path for LinkedIn post content (default: "linkedin_posts.json")
- `--agi-client`: AI client to use for generating posts. Options: 'local', 'openai' or 'anthropic' (default: 'local')
- `--model-name`: Name of the AI model to use (must be supported by chosen AGI client)

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

## Alternative Execution with Docker (Doesn't support Ollama)

1. Build Docker image
  ```bash
  docker build -t linkedin-post-gen . 
  ```
2. Run script in as container and output to disk
  ```bash
  docker run -e ANTHROPIC_API_KEY="YOUR_KEY" -v $(pwd)/output:/app/output linkedin-post-gen --agi-client anthropic --paper-url "https://arxiv.org/abs/2312.12456"
  ```

## Server Usage

1. Run Server
```bash
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
2. Access SwaggerUI at this url
```
http://127.0.0.1:8000/docs
```
3. Interact with the APIs using the UI