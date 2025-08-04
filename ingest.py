import fitz
import re
from langchain_community.embeddings import JinaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # You'll need this for the old chunker if you keep it for comparison
import json
from pinecone import Pinecone, ServerlessSpec
import os
import time
from tqdm.auto import tqdm
import math
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor # <-- ADD THIS LINE
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-policy-serverless-e5")
# Using a new index name for clarity
PDF_PATH = r"E:\hackathons\Bajaj Hackrx\2-stroma-rag\Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf"
JINA_API_KEY = os.environ.get("JINA_API_KEY")
# MODIFIED: Switched back to the Hugging Face model
EMBEDDING_MODEL = "jina-embeddings-v4"
DIMENSION = 2048 # MODIFIED: Dimension for e5-large-v2 is 1024

POSTGRES_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "database": os.environ.get("POSTGRES_DB", "rag_documents"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}
USE_SPARSE_VECTORS = False
sparse_encoder = None
embedding_model = JinaEmbeddings(
    model_name="jina-embeddings-v4",
    jina_api_key=JINA_API_KEY
)

def init_postgres_db():
    """Initialize PostgreSQL database and create necessary tables."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_hash VARCHAR(64) UNIQUE NOT NULL,
                content TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            );
        """)
        
        # Create document_chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                pinecone_id VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_pinecone_id ON document_chunks(pinecone_id);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("PostgreSQL database initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing PostgreSQL: {e}")
        return False

        
def get_db_connection():
    """Get a database connection."""
    return psycopg2.connect(**POSTGRES_CONFIG)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def calculate_file_hash(content):
    """Calculate MD5 hash of file content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def clause_chunker(document_text):
    """Splits a document based on section headers for better contextual grouping."""
    # This pattern looks for lines that start with numbering like 1., 1.1., (a), or (i)
    pattern = r'\n(?=\d+\.\d+\.|\d+\.|\([a-z]\)|\([ivx]+\))'
    clauses = re.split(pattern, document_text)
    
    final_chunks = []
    buffer = ""
    for clause in clauses:
        if not clause.strip():
            continue
        
        # Combine small clauses to form more meaningful chunks
        if len(buffer) + len(clause) < 200: # You can adjust this threshold
            buffer += clause + "\n\n"
        else:
            if buffer:
                final_chunks.append(buffer.strip())
            buffer = clause
    
    # Add the last remaining buffer to the chunks
    if buffer:
        final_chunks.append(buffer.strip())
        
    return final_chunks

def store_document_in_postgres(filename, content, metadata=None):
    """Store document in PostgreSQL and return document ID and a flag indicating if it was newly created."""
    try:
        file_hash = calculate_file_hash(content)
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if document already exists
        cursor.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
        existing_doc = cursor.fetchone()
        
        if existing_doc:
            print(f"Document {filename} already exists in database with ID: {existing_doc['id']}")
            cursor.close()
            conn.close()
            return existing_doc['id'], False  # Return False for 'is_new'
        
        # Insert new document
        cursor.execute("""
            INSERT INTO documents (filename, file_hash, content, metadata)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (filename, file_hash, content, json.dumps(metadata or {})))
        
        document_id = cursor.fetchone()['id']
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Document {filename} stored in PostgreSQL with ID: {document_id}")
        return document_id, True  # Return True for 'is_new'
        
    except Exception as e:
        print(f"Error storing document in PostgreSQL: {e}")
        return None, False

def store_chunks_in_postgres_and_pinecone(document_id, chunks, index):
    """
    Processes chunks in batches to create embeddings, then stores them
    in PostgreSQL and upserts the vectors to Pinecone.

    Args:
        document_id (int): The ID of the parent document from the 'documents' table.
        chunks (list): A list of text chunks to process.
        index: The initialized Pinecone index object.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        batch_size = 32  # Process 32 chunks at a time
        total_batches = math.ceil(len(chunks) / batch_size)

        print("Embedding and upserting data to Pinecone and PostgreSQL...")

        # Use tqdm to create a progress bar for the loop
        for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="Ingesting Chunks"):
            # 1. Select a batch of chunks
            i_end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:i_end]
            
            # 2. Generate embeddings for the batch using the Jina API
            dense_embeds = embedding_model.embed_documents(batch_chunks)

            # 3. Prepare data for Pinecone and PostgreSQL
            vectors_to_upsert = []
            for j, chunk_text in enumerate(batch_chunks):
                chunk_index = i + j
                pinecone_id = f"doc_{document_id}_chunk_{chunk_index}"

                # Prepare vector for Pinecone upsert
                vectors_to_upsert.append({
                    'id': pinecone_id,
                    'values': dense_embeds[j],
                    'metadata': {
                        'text': chunk_text,
                        'document_id': document_id,
                        'chunk_index': chunk_index
                    }
                })
                
                # Store chunk metadata in PostgreSQL
                cursor.execute(
                    """
                    INSERT INTO document_chunks (document_id, chunk_index, content, pinecone_id)
                    VALUES (%s, %s, %s, %s)
                    """, 
                    (document_id, chunk_index, chunk_text, pinecone_id)
                )

            # 4. Batch upsert to Pinecone
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert)

        # 5. Commit all the database transactions and close connections
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error storing chunks: {e}")
        # Rollback database changes on error
        if 'conn' in locals() and conn:
            conn.rollback()
        return False


# --- 3. MAIN INGESTION SCRIPT ---
if __name__ == "__main__":
    # Initialize the database and create tables if they don't exist
    init_postgres_db()

    if not os.path.exists(PDF_PATH):
        print(f"Error: Document not found at '{PDF_PATH}'")
        exit()

    print("Loading and processing PDF...")
    content = extract_text_from_pdf(PDF_PATH)
    if not content:
        print("Failed to extract text from PDF. Exiting.")
        exit()
    
    filename = os.path.basename(PDF_PATH)
    
    # Check if this document version has already been processed
    document_id, is_new = store_document_in_postgres(filename, content)
    if not is_new:
        print("This document has already been ingested. To re-ingest, please delete it via the API first.")
        exit()

    # Create or Recreate Pinecone Index if needed (optional, you can manage this manually)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        print(f"Creating new SERVERLESS index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
    
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone index is ready.")

    # Chunk the text
# NEW
    chunks = clause_chunker(content)
    print(f"Created {len(chunks)} chunks from document.")

    # Store chunks in PostgreSQL and vectors in Pinecone
    success = store_chunks_in_postgres_and_pinecone(document_id, chunks, index)
    if success:
        print("Ingestion complete. Document is now available for querying.")
    else:
        print("Ingestion failed.")