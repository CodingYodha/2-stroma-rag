
from dotenv import load_dotenv

load_dotenv()
import time
import json
import csv
import os
from langchain_core.documents import Document
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
import hashlib
from datetime import datetime
from langchain_community.embeddings import JinaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone
from langchain_core.documents import Document
import google.generativeai as genai
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever


import tempfile 
import asyncio


#==== 1. Configuration =====
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-policy-serverless-e5"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "database": os.environ.get("POSTGRES_DB", "rag_documents"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

EMBEDDING_MODEL = "jina-embeddings-v4"
DIMENSION = 2048
RERANKER_MODEL = 'BAAI/bge-reranker-large'

# Flag to enable/disable sparse vectors
USE_SPARSE_VECTORS = False

# Sample document path
SAMPLE_DOCUMENT_PATH = r"E:\Projects\hackrx_bajaj_finserv\BAJHLIP23020V012223.pdf"


JINA_API_KEY = os.environ.get("JINA_API_KEY")
embedding_model = JinaEmbeddings(
    model_name=EMBEDDING_MODEL,
    jina_api_key=JINA_API_KEY
)

def get_db_connection():
    """Get a database connection."""
    return psycopg2.connect(**POSTGRES_CONFIG)








# === 4. GLOBAL OBJECTS (Load models once at startup) ===
try:

    print("Loading document store...")
    with open("docstore.json", "r", encoding="utf-8") as f:
        docstore = json.load(f)
    print("Document store loaded successfully.")
    print("Loading models and connecting to services...")

    
    
    # Only initialize sparse encoder if we're using sparse vectors
    if USE_SPARSE_VECTORS:
        sparse_encoder = BM25Encoder.default()
    else:
        sparse_encoder = None
        print("WARNING: Sparse vectors disabled. Using dense-only search.")
    
    cohere_reranker = CohereRerank(cohere_api_key=COHERE_API_KEY,
                                    model="rerank-english-v3.0" )
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        raise NameError(f"Pinecone index '{PINECONE_INDEX_NAME}' not found.")

    index = pc.Index(PINECONE_INDEX_NAME)
    print("Models and services loaded successfully")
    

    
except Exception as e:
    print(f"Error during initialization: {e}")
    exit()


# === 5. Helper Functions and Classes ===
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        if isinstance(result, dict):
            result['time_taken_sec'] = round(end_time - start_time, 2)
        else:
            print(f"Executing {func.__name__}... took {end_time - start_time:.3f} seconds.")
        return result
    return wrapper

class SimilarityCache:
    def __init__(self, similarity_threshold=0.93):
        self.cache = []
        self.similarity_threshold = similarity_threshold

    def get(self, query_embedding):
        for cached_embedding, cached_answer in self.cache:
            similarity = np.dot(cached_embedding, query_embedding) / (
                np.linalg.norm(cached_embedding) * np.linalg.norm(query_embedding)
            )
            if similarity > self.similarity_threshold:
                print(f"---(Similarity Cache Hit! Score: {similarity:.4f})---")
                return cached_answer
        return None
    
    def add(self, query_embedding, answer):
        print("--- (Strong new answer in Similarity Cache) ---")
        self.cache.append((query_embedding, answer))

# Initialize the cache
similarity_cache = SimilarityCache()

def log_for_human_review(query, context, answer, feedback_status="unverified"):
    with open('hitl_feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Timestamp', 'Query', 'Context', 'Answer', 'Feedback'])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), query, context, answer, feedback_status])

def get_chunk_details_from_postgres(pinecone_ids):
    """Get additional chunk details from PostgreSQL using Pinecone IDs."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        placeholders = ','.join(['%s'] * len(pinecone_ids))
        query = f"""
            SELECT dc.pinecone_id, dc.content, d.filename, d.metadata
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.pinecone_id IN ({placeholders})
        """
        
        cursor.execute(query, pinecone_ids)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {row['pinecone_id']: row for row in results}
        
    except Exception as e:
        print(f"Error getting chunk details from PostgreSQL: {e}")
        return {}


# === 6. CORE RAG PIPELINE FUNCTIONS ===
@timing_decorator
def retrieve_chunks(query, query_embedding, top_k=5):
    """
    Retrieve relevant PARENT chunks.
    1. Query Pinecone for the most relevant CHILD chunks.
    2. Get the parent_id from the metadata of the child chunks.
    3. Use the docstore to look up the full text of the parent chunks.
    """
    try:
        # 1. Query Pinecone for child chunks
        results = index.query(
            vector=query_embedding,
            top_k=top_k, # Retrieve a few child chunks to find unique parents
            include_metadata=True
        )
        
        # 2. Get unique parent IDs from the results
        unique_parent_ids = set()
        for match in results['matches']:
            if match['metadata'].get('parent_id'):
                unique_parent_ids.add(match['metadata']['parent_id'])

        # 3. Look up parent chunks from the docstore
        parent_chunks = []
        for pid in unique_parent_ids:
            if pid in docstore:
                # Find the original filename and doc_id from the first match
                original_metadata = next((m['metadata'] for m in results['matches'] if m['metadata'].get('parent_id') == pid), {})
                
                parent_chunks.append({
                    'text': docstore[pid],
                    'filename': original_metadata.get('filename'),
                    'document_id': original_metadata.get('document_id')
                })
        
        return parent_chunks
    
    except Exception as e:
        print(f"Error in retrieve_chunks: {e}")
        return []


@timing_decorator
def generate_final_answer(context, question, source_info=None):
    print("Generating final answer with Google Gemini 1.5 Flash...")
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    source_context = f"\n\nSource: {source_info.get('filename', 'Unknown')}" if source_info else ""
    
    prompt = f"""
**Role:** You are an expert AI assistant specialized in accurately interpreting documents.

**Task:** Your task is to answer the user's question based ONLY on the provided context below.

**Instructions:**
1. **Stay Grounded:** Base your answer strictly on the information within the provided 'Context'. Do not use any external knowledge or make assumptions.
2. **Be Precise:** Synthesize the information to form a complete, clear, and helpful sentence. Do not just copy-paste phrases from the context.
3. **Handle Negatives:** Pay close attention to keywords that indicate conditions or exclusions, such as "unless," "except," "not covered," or section headings that say "Exclusion." If the context describes an exclusion, your answer must clearly state that it is not covered.
4. **Address Missing Information:** If the context does not contain enough information to answer the question, you must explicitly state that the answer is not available in the provided text.

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
        response = model.generate_content(prompt)
        return {
            "answer": response.text.strip(), 
            "source_context": context,
            "source_info": source_info or {}
        }
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return {
            "answer": "There was an error generating the answer.", 
            "source_context": context,
            "source_info": source_info or {}
        }


# === 7. FASTAPI DATA MODELS ===
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the waiting period for pre-existing diseases?")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    source_context: str
    source_filename: Optional[str] = None
    total_time_taken_sec: float

class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., example=['Question 1?', 'Question 2?'])

class BatchAnswerResponse(BaseModel):
    answers: List[dict]

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    chunks_created: int


# === 8. FASTAPI APPLICATION ===
app = FastAPI(
    title="Enhanced RAG API with PostgreSQL",
    description="An API to query documents using PostgreSQL for storage and Pinecone for vector search."
)

@app.get("/")
def root():
    return {
        "message": "Enhanced RAG API with PostgreSQL and Pinecone", 
        "sparse_vectors_enabled": USE_SPARSE_VECTORS,
        "database": "PostgreSQL + Pinecone",
        "docs": "/docs"
    }

@app.get("/documents")
def list_documents():
    """List all documents in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT d.id, d.filename, d.upload_date, d.metadata,
                   COUNT(dc.id) as chunk_count
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            GROUP BY d.id, d.filename, d.upload_date, d.metadata
            ORDER BY d.upload_date DESC
        """)
        
        documents = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")



@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new document."""
    try:
        content = await file.read()

        if file.filename.lower().endswith('.pdf'):
            # Use a cross-platform temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            text_content = extract_text_from_pdf(temp_path)
            os.remove(temp_path)  # Clean up the temporary file
        else:
            text_content = content.decode('utf-8')
        
        # ... (rest of your function remains the same) ...

        if not text_content:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        metadata = {
            "upload_source": "api",
            "content_type": file.content_type,
            "processed_at": datetime.now().isoformat()
        }
        
        document_id, is_new = store_document_in_postgres(file.filename, text_content, metadata)
        if not document_id:
            raise HTTPException(status_code=500, detail="Failed to store document")
        
        if not is_new:
            # Document already existed
            return DocumentUploadResponse(
                message="Document already exists in the database. No new data was added.",
                document_id=document_id,
                chunks_created=0
            )

        chunks = chunk_text(text_content)
        success = store_chunks_in_postgres_and_pinecone(document_id, chunks)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process document chunks")
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_id,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {e}")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QueryRequest):
    total_start_time = time.time()

    # 1. Embedding the query
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

    # 3. Retrieve initial chunks from Pinecone
    retrieved_chunks = retrieve_chunks(request.query, query_embedding, top_k=10)
    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="Could not find any relevant context for the query")
    
    # 4. Rerank with Cohere API (CORRECTED)
    print("Reranking context with Cohere API...")
    
    # Create a list of Document objects instead of plain strings
    documents_to_rerank = [Document(page_content=chunk['text'], metadata=chunk) for chunk in retrieved_chunks]
    
    # Pass the Document objects to the reranker
    reranked_docs = cohere_reranker.compress_documents(
        documents=documents_to_rerank,
        query=request.query
    )

    if not reranked_docs:
        raise HTTPException(status_code=404, detail="Reranking did not find a relevant context.")

    # The best context and its source info now come from the first reranked Document object
    best_context = reranked_docs[0].page_content
    source_info = reranked_docs[0].metadata

    # 5. Generate Final answer
    final_answer_result = generate_final_answer(best_context, request.query, source_info)

    # 6. Add to cache and log to HITL
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

@app.post("/process-batch", response_model=BatchAnswerResponse)
def process_batch_questions(request: BatchQueryRequest):
    all_answers = []
    for query in request.questions:
        total_start_time = time.time()
        final_answer_result = {}

        # 1. Embed query
        query_embedding = embedding_model.embed_query(query)
        
        # 2. Check Cache
        cached_result = similarity_cache.get(query_embedding)

        if cached_result:
            final_answer_result = cached_result
        else:
            # 3. Retrieve initial chunks
            retrieved_chunks = retrieve_chunks(query, query_embedding, top_k=10)
            
            if not retrieved_chunks:
                final_answer_result = {"answer": "Could not find any relevant context for the query.", "source_context": "", "source_info": {}}
            else:
                # 4. Rerank with Cohere API (The Corrected Logic)
                print(f"Reranking context for batch query: '{query}'...")
                
                # Create a list of Document objects from the retrieved chunks
                documents_to_rerank = [Document(page_content=chunk['text'], metadata=chunk) for chunk in retrieved_chunks]
                
                # Pass the list of Document objects to the reranker
                reranked_docs = cohere_reranker.compress_documents(
                    documents=documents_to_rerank,
                    query=query
                )

                if not reranked_docs:
                    final_answer_result = {"answer": "Reranking did not find a relevant context.", "source_context": "", "source_info": {}}
                else:
                    # Get the best context and source info from the reranked Document object
                    best_context = reranked_docs[0].page_content
                    source_info = reranked_docs[0].metadata
                    
                    # 5. Generate Final answer
                    final_answer_result = generate_final_answer(best_context, query, source_info)
                    
                    # 6. Add to cache and log to HITL
                    similarity_cache.add(query_embedding, final_answer_result)
                    log_for_human_review(query, best_context, final_answer_result.get("answer", ""))

        total_time_taken = time.time() - total_start_time

        all_answers.append({
            "query": query,
            "answer": final_answer_result.get('answer'),
            "source_context": final_answer_result.get('source_context'),
            "source_filename": final_answer_result.get('source_info', {}).get('filename'),
            "total_time_taken_sec": round(total_time_taken, 2)
        })

    return BatchAnswerResponse(answers=all_answers)

@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    """Delete a document and its associated chunks."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get chunk IDs for Pinecone deletion
        cursor.execute("SELECT pinecone_id FROM document_chunks WHERE document_id = %s", (document_id,))
        chunk_ids = [row['pinecone_id'] for row in cursor.fetchall()]
        
        # Delete from Pinecone
        if chunk_ids:
            index.delete(ids=chunk_ids)
            print(f"Deleted {len(chunk_ids)} vectors from Pinecone")
        
        # Delete from PostgreSQL (cascades to chunks)
        cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} and {len(chunk_ids)} chunks deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)