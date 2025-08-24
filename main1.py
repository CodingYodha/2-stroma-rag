# main.py

from dotenv import load_dotenv
load_dotenv()

import time
import os
import json
import csv
import re
import tempfile
import hashlib
from typing import List, Optional

import fitz # PyMuPDF
import numpy as np
import psycopg2
import uvicorn
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from loguru import logger
from pinecone import Pinecone, ServerlessSpec
from psycopg2.extras import RealDictCursor
from tqdm.auto import tqdm

from langchain_core.documents import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-policy-serverless-e5"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "database": os.environ.get("POSTGRES_DB", "rag_documents"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

EMBEDDING_MODEL_NAME = "jina-embeddings-v4"
DIMENSION = 2048 # jina-v2 is 768, jina-v4 is 2048. Ensure this matches your model and Pinecone index.

# ==============================================================================
# 2. HELPER FUNCTIONS & CLASSES
# (These are helper functions that need to be defined for the upload endpoint to work)
# You will need to provide the implementation for these based on your original project structure.
# ==============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        logger.success(f"Successfully extracted text from '{pdf_path}'.")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    return psycopg2.connect(**POSTGRES_CONFIG)

def store_document_in_postgres(filename: str, content: str, metadata: dict) -> (int, bool):
    """Stores document metadata in PostgreSQL and returns the document ID and a flag indicating if it's new."""
    file_hash = hashlib.sha256(content.encode()).hexdigest()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
    existing_doc = cursor.fetchone()
    
    if existing_doc:
        logger.info(f"Document '{filename}' with hash {file_hash[:10]}... already exists with ID {existing_doc[0]}.")
        cursor.close()
        conn.close()
        return existing_doc[0], False # Not new

    # CORRECTED SQL STATEMENT: Added the 'content' column and its corresponding value.
    cursor.execute(
        "INSERT INTO documents (filename, file_hash, content, metadata) VALUES (%s, %s, %s, %s) RETURNING id",
        (filename, file_hash, content, json.dumps(metadata)) # Added the 'content' variable here
    )
    document_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    logger.success(f"Stored new document '{filename}' in PostgreSQL with ID {document_id}.")
    return document_id, True # Is new
def clause_chunker(document_text: str) -> list[str]:
    """Splits document text into clauses based on common legal/policy document patterns."""
    pattern = r'\n(?=\d+\.\d+\.|\d+\.|\([a-z]\)|\([ivx]+\))'
    return [clause for clause in re.split(pattern, document_text) if clause.strip()]

def chunk_text(text_content: str) -> List[Document]:
    """Chunks the text content using parent-child logic."""
    parent_chunks_text = clause_chunker(text_content)
    parent_documents = [Document(page_content=text) for text in parent_chunks_text]
    
    # This is your 'docstore' logic
    docstore = {f"parent_{i}": doc.page_content for i, doc in enumerate(parent_documents)}

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    child_documents = child_splitter.split_documents(parent_documents)

    for i, doc in enumerate(child_documents):
        # Find which parent this child belongs to
        for parent_id, parent_content in docstore.items():
            if doc.page_content in parent_content:
                doc.metadata['parent_id'] = parent_id
                break
    
    return child_documents

def store_chunks_in_postgres_and_pinecone(document_id: int, chunks: List[Document]):
    """Stores text chunks in PostgreSQL and their vector embeddings in Pinecone."""
    conn = get_db_connection()
    cursor = conn.cursor()

    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and Upserting Chunks"):
        batch_end = min(i + batch_size, len(chunks))
        batch_docs = chunks[i:batch_end]

        texts_to_embed = [doc.page_content for doc in batch_docs]
        embeds = embedding_model.embed_documents(texts_to_embed)

        pinecone_vectors = []
        for k, doc in enumerate(batch_docs):
            # This combination creates a unique index for each chunk
            chunk_index = i + k
            pinecone_id = f"{document_id}-{chunk_index}"
            doc.metadata['document_id'] = document_id

            # MODIFIED: Added 'chunk_index' to the SQL statement and parameters
            cursor.execute(
                """
                INSERT INTO document_chunks (document_id, pinecone_id, content, metadata, chunk_index)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (document_id, pinecone_id, doc.page_content, json.dumps(doc.metadata), chunk_index)
            )

            # Prepare vector for Pinecone
            pinecone_vectors.append({
                'id': pinecone_id,
                'values': embeds[k],
                'metadata': doc.metadata
            })

        index.upsert(vectors=pinecone_vectors)

    conn.commit()
    cursor.close()
    conn.close()
    logger.success(f"Successfully stored and indexed {len(chunks)} chunks for document ID {document_id}.")
    return True

def timing_decorator(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        if isinstance(result, dict):
            result['time_taken_sec'] = time_taken
        else:
            logger.info(f"Function {func.__name__} took {time_taken} seconds.")
        return result
    return wrapper

class SimilarityCache:
    """A simple cache to store answers for similar questions."""
    def __init__(self, similarity_threshold=0.95):
        self.cache = []
        self.similarity_threshold = similarity_threshold

    def get(self, query_embedding):
        for cached_embedding, cached_answer in self.cache:
            similarity = np.dot(cached_embedding, query_embedding) / (np.linalg.norm(cached_embedding) * np.linalg.norm(query_embedding))
            if similarity > self.similarity_threshold:
                logger.info(f"--- Similarity Cache Hit! Score: {similarity:.4f} ---")
                return cached_answer
        return None
    
    def add(self, query_embedding, answer):
        logger.info("--- Storing new answer in Similarity Cache ---")
        self.cache.append((query_embedding, answer))

def log_for_human_review(query, context, answer, feedback_status="unverified"):
    """Logs query-answer pairs to a CSV for human-in-the-loop feedback."""
    file_exists = os.path.isfile('hitl_feedback.csv')
    with open('hitl_feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Query', 'Context', 'Answer', 'Feedback'])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), query, context, answer, feedback_status])

# ==============================================================================
# 3. GLOBAL OBJECTS (Load models once at startup)
# ==============================================================================
try:
    logger.info("Loading models and connecting to services...")

    embedding_model = JinaEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        jina_api_key=JINA_API_KEY
    )
    
    cohere_reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        raise NameError(f"Pinecone index '{PINECONE_INDEX_NAME}' not found.")
    index = pc.Index(PINECONE_INDEX_NAME)

    similarity_cache = SimilarityCache()
    
    # Loading the parent document store from PostgreSQL would be more robust,
    # but for now, we'll retrieve parent context directly during the query phase.
    
    logger.success("Models and services loaded successfully.")
    
except Exception as e:
    logger.critical(f"Error during initialization: {e}")
    exit()

# ==============================================================================
# 4. CORE RAG PIPELINE
# ==============================================================================
@timing_decorator
def retrieve_chunks(query_embedding, top_k=15):
    """Retrieve relevant child chunks from Pinecone."""
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
    except Exception as e:
        logger.error(f"Error retrieving chunks from Pinecone: {e}")
        return []

def get_parent_context(child_chunks: list) -> list:
    """
    Retrieves the full parent context for a list of child chunks.
    This version queries PostgreSQL for reliability.
    """
    parent_ids = list(set([match['metadata'].get('parent_id') for match in child_chunks if 'parent_id' in match['metadata']]))
    
    if not parent_ids:
        # Fallback to using child chunks if no parent_ids are found
        return [Document(page_content=chunk['metadata']['text'], metadata=chunk['metadata']) for chunk in child_chunks]

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # This is a simplified approach. A more robust solution might involve another table for parent chunks.
        # For now, we assume the parent content can be reconstructed or is stored with the child.
        # Let's get the original documents associated with the parents.
        doc_ids = list(set([match['metadata'].get('document_id') for match in child_chunks]))

        placeholders = ','.join(['%s'] * len(doc_ids))
        query = f"SELECT content, filename FROM documents WHERE id IN ({placeholders})"
        cursor.execute(query, doc_ids)
        
        # This is a placeholder for a more complex parent-retrieval logic.
        # For now, let's just use the child chunks' text as context, which is a common and effective strategy.
        parent_docs = [Document(page_content=chunk['metadata']['text'], metadata=chunk['metadata']) for chunk in child_chunks]

        cursor.close()
        conn.close()
        return parent_docs
        
    except Exception as e:
        logger.error(f"Error getting parent context from PostgreSQL: {e}")
        return [] # Return empty list on error

@timing_decorator
def generate_final_answer(context: str, question: str, source_info: dict = None):
    """Generates a final answer using Google Gemini based on the provided context."""
    logger.info("Generating final answer with Google Gemini...")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    source_context = f"\n\nSource Document: {source_info.get('filename', 'Unknown')}" if source_info else ""
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    prompt = f"""
        **Role:** You are an expert AI assistant specialized in accurately interpreting policy documents.
        **Task:** Answer the user's question based ONLY on the provided context.
        **Instructions:**
        1. **Stay Grounded:** Base your answer strictly on the information within the 'Context'. Do not use external knowledge.
        2. **Be Precise:** Synthesize the information into a clear and helpful answer.
        3. **Handle Negatives:** If the context describes an exclusion or condition (e.g., "not covered," "unless," "except"), state it clearly.
        4. **Address Missing Information:** If the answer is not in the context, state that the information is not available in the provided text.
        ---
        **## Context**
        {context}{source_context}
        ---
        **## Question**
        {question}
        ---
        **Helpful Answer:**
    """
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        if not response.parts:
            raise ValueError("Response was empty, possibly blocked by API safety filters.")
        return {
            "answer": response.text.strip(), 
            "source_context": context,
            "source_info": source_info or {}
        }
    except Exception as e:
        logger.exception(f"Error during Gemini API call for question '{question}': {e}")
        return {
            "answer": "There was an error generating the answer.", 
            "source_context": context,
            "source_info": source_info or {}
        }

# ==============================================================================
# 5. FASTAPI DATA MODELS
# ==============================================================================
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the waiting period for pre-existing diseases?")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    source_context: str
    source_filename: Optional[str] = None
    total_time_taken_sec: float

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    chunks_created: int

# ==============================================================================
# 6. FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(
    title="Stateful RAG API",
    description="An API to upload documents and ask questions against a persistent vector store."
)

@app.get("/")
def root():
    return {
        "message": "Stateful RAG API is running.", 
        "docs_url": "/docs",
        "workflow": "1. POST /upload-document to add a PDF. 2. POST /ask to query."
    }

@app.post("/upload-document", response_model=DocumentUploadResponse, tags=["Document Management"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document, storing it for future queries."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        text_content = extract_text_from_pdf(temp_path)
        os.remove(temp_path)

        if not text_content:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        
        metadata = {"upload_source": "api", "content_type": file.content_type}
        
        document_id, is_new = store_document_in_postgres(file.filename, text_content, metadata)
        if not is_new:
            return DocumentUploadResponse(
                message="Document with this content already exists.",
                document_id=document_id,
                chunks_created=0
            )

        chunks = chunk_text(text_content)
        store_chunks_in_postgres_and_pinecone(document_id, chunks)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully.",
            document_id=document_id,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        logger.exception("Error during document upload.")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {e}")

@app.post("/ask", response_model=AnswerResponse, tags=["Querying"])
def ask_question(request: QueryRequest):
    """
    Ask a single question. The API will find the most relevant document
    in the database and generate an answer.
    """
    total_start_time = time.time()

    # 1. Embed the query
    query_embedding = embedding_model.embed_query(request.query)

    # 2. Check Cache
    cached_result = similarity_cache.get(query_embedding)
    if cached_result:
        total_time_taken = time.time() - total_start_time
        return AnswerResponse(
            query=request.query,
            answer=cached_result['answer'],
            source_context=cached_result['source_context'],
            source_filename=cached_result.get('source_info', {}).get('filename'),
            total_time_taken_sec=round(total_time_taken, 2)
        )

    # 3. Retrieve initial child chunks from Pinecone
    retrieved_child_chunks = retrieve_chunks(query_embedding, top_k=15)
    if not retrieved_child_chunks:
        raise HTTPException(status_code=404, detail="Could not find any relevant context for the query.")
    
    # 4. Create Document objects for reranking
    documents_to_rerank = [
        Document(page_content=chunk['metadata']['text'], metadata=chunk['metadata'])
        for chunk in retrieved_child_chunks if 'text' in chunk.get('metadata', {})
    ]

    # 5. Rerank with Cohere
    logger.info(f"Reranking {len(documents_to_rerank)} chunks with Cohere...")
    reranked_docs = cohere_reranker.compress_documents(
        documents=documents_to_rerank,
        query=request.query
    )

    if not reranked_docs:
        raise HTTPException(status_code=404, detail="Reranking did not find a relevant context.")

    best_context = reranked_docs[0].page_content
    source_info = reranked_docs[0].metadata

    # 6. Generate Final answer
    final_answer_result = generate_final_answer(best_context, request.query, source_info)

    # 7. Add to cache and log for human review
    similarity_cache.add(query_embedding, final_answer_result)
    log_for_human_review(request.query, best_context, final_answer_result["answer"])

    total_time_taken = time.time() - total_start_time

    return AnswerResponse(
        query=request.query,
        answer=final_answer_result['answer'],
        source_context=final_answer_result['source_context'],
        source_filename=source_info.get('filename'),
        total_time_taken_sec=round(total_time_taken, 2)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)