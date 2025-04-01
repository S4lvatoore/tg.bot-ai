import logging
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.ext import ContextTypes
import threading
import uvicorn
import os

# ===========================
# Connect to Milvus
# ===========================
host = "your-host"  # Replace with your Milvus server host
port = "your-port"  # Replace with your Milvus server port
user = "your-username"  # Replace with your Milvus username
password = "your-password"  # Replace with your Milvus password
# Connect to Milvus server

connections.connect(
    alias="default",
    host=host,
    port=port,
    user=user,
    password=password,
    secure=True
)
print("Connected to Milvus")

# ===========================
# Set up Milvus collection
# ===========================
collection_name = "articles"

# Check if collection exists and drop if necessary
if collection_name in utility.list_collections():
    collection = Collection(collection_name)
    collection.drop()  
    print(f"Collection '{collection_name}' deleted.")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]

# Create schema and collection in Milvus
schema = CollectionSchema(fields, description="Vector DB for articles")
collection = Collection(collection_name, schema)
print(f"Collection '{collection_name}' created.")

# ===========================
# Embedding model
# ===========================
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===========================
# FastAPI application
# ===========================
app = FastAPI()

class Query(BaseModel):
    query_text: str
    top_k: int = 5

class FilePath(BaseModel):
    file_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the bot server!"}

# ===========================
# Telegram bot integration
# ===========================
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Ask me anything about AI.')

# Function to handle messages and search
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text  
    query_vector = hf_model.encode([user_input]).tolist()  
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    try:
        collection.create_index(field_name="embedding", index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        })
        print("Index created successfully.")
    except Exception as e:
        print(f"Error while creating index: {str(e)}")

    collection.load()

    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=1,  
        output_fields=["title", "content"]
    )

    response = "Here are the most relevant articles:\n"
    for result in results[0]:
        title = result.entity.get('title')
        content = result.entity.get('content')
        response += f"Title: {title}\nContent: {content}\n\n"
    
    await update.message.reply_text(response)

# ===========================
# Endpoints for uploading data from CSV
# ===========================

@app.post("/insert_data")
async def insert_data(data: FilePath):
    try:
        file_path = data.file_path
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        df = pd.read_csv(file_path)

        if 'title' not in df.columns or 'content' not in df.columns:
            return {"error": "CSV file must contain 'title' and 'content' columns"}

        titles = df['title'].tolist()
        contents = df['content'].tolist()

        # Generate embeddings for each content
        embeddings = hf_model.encode(contents).tolist()

        # Prepare data for insertion
        data_to_insert = [titles, contents, embeddings]
        collection.insert(data_to_insert, fields=["title", "content", "embedding"])
        
        return {"message": f"Inserted {len(df)} articles into Milvus."}

    except Exception as e:
        return {"error": str(e)}

# Telegram bot main function
def main():
    application = Application.builder().token("your-token").build()

    # Set up the command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))
    application.run_polling()

if __name__ == "__main__":
    threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000)).start()
    main()
