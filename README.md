# Milvus-based Article Search Bot

## Overview
This project is a Python application that integrates **Milvus**, **FastAPI**, and a **Telegram Bot** for vector-based article search. The app allows you to:
- Connect to a Milvus server (local or cloud).
- Manage a collection of articles.
- Generate embeddings (vector representations) for articles using the `SentenceTransformer` model ("all-MiniLM-L6-v2").
- Perform semantic searches based on vector similarity.
- Upload data from a CSV file.
- Interact with the system through a Telegram bot.

## Prerequisites
- **Telegram Bot:**  
  Before starting, create a Telegram bot using [BotFather](https://core.telegram.org/bots#6-botfather) and obtain your bot token.
- **Milvus Server:**  
  You must have access to a Milvus server. For cloud deployments, proper authentication credentials are required.
- **CSV File Requirements:**  
  When uploading data via the API, the CSV file path must be provided in the following JSON format:
  ```json
  {
    "file_path": {
      "file_path": "C:\\Users\\path\\to\\your\\csv"
    }
  }
