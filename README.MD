# Summarizer Chat Bot

This project is a chat bot built with [LangChain](https://github.com/langchain-ai/langchain) that summarizes content from various document formats, including PDF, DOCX, and HTML files. The primary goal of this project is to learn and experiment with LangChain's capabilities for document processing and summarization.

## Features

- **Multi-format support:** Accepts PDF, DOCX, and HTML files as input.
- **Automated summarization:** Extracts and summarizes the main points from the provided documents.
- **Conversational interface:** Interact with the bot to upload documents and receive concise summaries.

## How It Works

1. Upload a document (PDF, DOCX, or HTML).
2. The bot processes and extracts the text.
3. LangChain is used to generate a summary of the content.
4. The summary is returned in the chat interface.

## Requirements

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- Libraries for document parsing (e.g., `PyPDF2`, `python-docx`, `BeautifulSoup4`)

## Getting Started

1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the FastAPI server with Uvicorn:
    ```bash
    uvicorn core.app:app --reload
    ```
4. Access the chat bot via your browser at `http://localhost:8000`.

