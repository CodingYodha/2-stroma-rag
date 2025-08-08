# Stroma-RAG: A Production-Grade, Dual-Mode RAG Application

This project is a complete, full-stack Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware answers from complex documents. It was developed to overcome the limitations of traditional keyword search by leveraging state-of-the-art AI models and a sophisticated, multi-stage processing pipeline.

The application features a dual architecture:
1.  A **stateful RAG service** with a persistent database for building a permanent, multi-document knowledge base.
2.  A **stateless, on-the-fly webhook** designed for scenarios like hackathons, where documents must be processed and queried in a single, isolated transaction.

## Key Features

* **Advanced RAG Pipeline:** Implements a state-of-the-art retrieve-rerank-generate pipeline for maximum accuracy.
* **Semantic Parent/Child Chunking:** Utilizes an advanced chunking strategy where documents are split into large, semantic parent chunks for context and small child chunks for precise vector retrieval.
* **Best-of-Breed API Integration:** Leverages a powerful multi-API stack:
    * **Embeddings:** Jina AI Embeddings v4
    * **Vector Search:** Pinecone Serverless
    * **Reranking:** Cohere Rerank v3
    * **Generation:** Google Gemini Pro
* **Production-Ready Backend:** Built with FastAPI, featuring asynchronous processing, Pydantic for robust data validation, and Loguru for structured logging.
* **Triple Storage Strategy:** A robust architecture using PostgreSQL as the primary source of truth, Pinecone for vector storage, and a JSON `docstore` for parent chunk mapping.
* **Performance Optimization:** Includes an in-memory semantic cache to provide instant responses for repeated or similar queries.
* **Stateless Webhook:** A dedicated, self-contained API endpoint that can process a new document from a URL on the fly, using a temporary Pinecone index that is automatically created and deleted for each request.
* **Deployment:** Fully configured for cloud deployment on platforms like Railway.

## Architecture & Data Flow

The system is composed of two distinct pipelines: an Ingestion Pipeline to process and store documents, and a Query Pipeline to answer user questions.

<details>
<summary>Click to view the Data Flow Diagram</summary>

```mermaid
graph TD
    subgraph Ingestion Pipeline
        A[Upload Document] --> B(Chunking);
        B --> C(Embed Child Chunks);
        B --> D(Store Parent Chunks);
        B --> E(Store Document Metadata);
        C --> F[(Vector DB - Pinecone)];
        D --> H{{docstore.json}};
        E --> G[(Relational DB - PostgreSQL)];
    end

    subgraph Query Pipeline
        Q1[User Query] --> Q2(Embed Query);
        Q2 --> Q3(Vector Search for Child Chunks);
        Q3 --> Q4(Retrieve Parent Chunks);
        Q4 --> Q5(Rerank Parent Chunks);
        Q5 --> Q6(Build Prompt);
        Q6 --> Q7[LLM Response];
    end
```
</details>

## Tech Stack

* **Backend:** Python, FastAPI
* **AI / LangChain:** LangChain, Jina API, Pinecone, Cohere API, Google Gemini API
* **Database:** PostgreSQL
* **Deployment:** Railway, Uvicorn
* **Testing & Logging:** Postman, Loguru

## Local Setup and Installation

Follow these steps to run the application locally.

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd 2-stroma-rag
```

**2. Create and Activate a Virtual Environment**
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Activate venv (macOS/Linux)
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**
Create a file named `.env` in the root directory and populate it with your credentials. Use the `.env.example` as a template:

**.env.example**
```
# API Keys
PINECONE_API_KEY="your_pinecone_key_here"
JINA_API_KEY="your_jina_key_here"
COHERE_API_KEY="your_cohere_key_here"
GOOGLE_API_KEY="your_google_ai_key_here"

# Database Connection URL (for local PostgreSQL)
DATABASE_URL="postgresql://postgres:your_password@localhost:5432/rag_documents"

# For ingest.py script
PDF_PATH="path/to/your/document.pdf"
PINECONE_INDEX_NAME="rag-policy-parent-child"
```

**5. Run the Ingestion Script**
To populate the stateful part of the application, run the ingestion script. This will process your PDF, populate PostgreSQL, create the Pinecone index, and generate the `docstore.json` file.
```bash
python ingest.py
```

**6. Run the FastAPI Server**
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## API Usage

The application exposes several endpoints. The main endpoint for the hackathon is the stateless webhook.

| Method | Path                           | Description                                                                 |
| :----- | :----------------------------- | :-------------------------------------------------------------------------- |
| **POST** | `/api/v1/hackrx/run`           | **Stateless Webhook.** Processes a PDF from a URL and answers questions.  |
| **POST** | `/ask`                         | Queries the pre-ingested (stateful) knowledge base with a single question.  |
| **POST** | `/process-batch`               | Queries the stateful knowledge base with a list of questions.               |
| **GET** | `/documents`                   | Lists all documents in the persistent PostgreSQL database.                |
| **POST** | `/upload-document`             | Uploads a new document to the persistent knowledge base.                  |
| **DELETE** | `/documents/{document_id}`     | Deletes a document from the persistent knowledge base.                      |

### Example: Testing the Stateless Webhook

You can test the primary webhook using Postman or a command-line tool like `curl`.

```bash
curl -X POST \
  [http://127.0.0.1:8000/api/v1/hackrx/run](http://127.0.0.1:8000/api/v1/hackrx/run) \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "[https://your-public-url-to-a-pdf.pdf](https://your-public-url-to-a-pdf.pdf)",
    "questions": ["What is the main topic of this document?"]
}'
```

## Acknowledgments

Special thanks to my teammate **Aryan Karmore** for architecting and implementing the entire PostgreSQL integration, which serves as the foundational persistence layer for this application.
