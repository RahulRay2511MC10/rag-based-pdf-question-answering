
# AI-Powered Multi-Document RAG Question Answering System

A Retrieval-Augmented Generation (RAG) based PDF question-answering system that allows users to upload and query multiple PDF documents through an interactive Streamlit interface.

The system combines semantic retrieval, FAISS vector search, Cross-Encoder reranking, query rewriting, source-level citations, and answer evaluation to provide more reliable and traceable answers.

---

## Features

- Upload and process multiple PDF documents
- Search across all uploaded documents
- Search within a specific PDF
- PDF text extraction and chunking
- Configurable chunk size and chunk overlap
- OpenAI `text-embedding-3-small` embeddings
- FAISS vector similarity search
- Cross-Encoder reranking
- Query rewriting
- LLM-based answer generation
- Source and page-level citations
- Retrieved evidence visualization
- Retrieved-context inspection
- RAG pipeline trace
- Semantic similarity evaluation
- LLM-as-a-Judge evaluation
- Correctness, relevance, and faithfulness scoring
- Interactive Streamlit interface

---

## System Architecture

```text
                    PDF Documents
                          |
                          v
                  PDF Text Extraction
                          |
                          v
                    Text Chunking
                          |
                          v
              OpenAI Embeddings
          (text-embedding-3-small)
                          |
                          v
                  FAISS Vector Store
                          |
                          v
                    User Query
                          |
                          v
                    Query Rewriting
                          |
                          v
                  Semantic Retrieval
                          |
                          v
                 Source Filtering
                          |
                          v
                Cross-Encoder Reranker
                          |
                          v
                  Top-K Context
                          |
                          v
                 LLM Answer Generation
                          |
                          v
              Answer + Source Citations
                          |
                          v
                     Evaluation
```

---

## RAG Pipeline

The application follows the following pipeline:

### 1. PDF Upload

Users can upload one or multiple PDF documents through the Streamlit interface.

Each document retains metadata such as:

- Source filename
- Page number
- Chunk identifier

This metadata is used later for document filtering and citations.

### 2. PDF Processing

The uploaded PDFs are parsed and their textual content is extracted.

The extracted documents are then divided into smaller chunks.

Default configuration:

```text
Chunk Size: 500
Chunk Overlap: 100
```

The chunk size and overlap can be configured from the Streamlit sidebar.

### 3. Embedding Generation

Each chunk is converted into a dense vector representation using:

```text
text-embedding-3-small
```

The same embedding model is used for generating the query embedding.

### 4. FAISS Retrieval

The document embeddings are stored in a FAISS vector index.

The system uses:

```text
FAISS IndexFlatL2
```

The user's query is embedded and compared against the stored document vectors using L2 distance.

The top candidate chunks are retrieved.

### 5. Document Filtering

The application supports two search modes.

#### All Documents

```text
Search in → All Documents
```

The query can retrieve information from any uploaded PDF.

#### Specific Document

```text
Search in → Rahul.pdf
```

Only chunks belonging to the selected PDF are considered for the query.

This is possible because every chunk retains its source filename in its metadata.

### 6. Cross-Encoder Reranking

The initially retrieved candidates are passed through a Cross-Encoder reranker.

The reranker evaluates the relationship between:

```text
Query + Retrieved Chunk
```

and assigns a relevance score.

The candidates are then reordered according to their reranking scores.

This provides a second-stage retrieval process:

```text
FAISS Retrieval
      ↓
Candidate Chunks
      ↓
Cross-Encoder
      ↓
Reranked Chunks
      ↓
Most Relevant Context
```

### 7. Query Rewriting

The system can rewrite the user's query before retrieval.

This is particularly useful for conversational questions where the current question depends on previous conversation context.

For example:

```text
User:
What is his CGPA?

Conversation:
Who is Rahul?
Rahul is an M.Tech student at IIT Patna.

Rewritten query:
What is Rahul's CGPA?
```

The rewritten query can then be used for retrieval.

### 8. Context Construction

The highest-ranked chunks are combined into the context supplied to the LLM.

Each retrieved chunk retains its document and page metadata.

Example:

```text
[Source: Rahul.pdf | Page: 2]

Retrieved text...
```

### 9. Answer Generation

The retrieved context is passed to the LLM.

The model generates an answer based on the retrieved document context.

### 10. Citations

The application displays source-level citations for the retrieved information.

Example:

```text
Rahul.pdf — Page 2
Research_Paper.pdf — Page 5
```

This allows users to trace the generated answer back to the source document.

---

# Evaluation

The application includes an evaluation interface for measuring generated-answer quality.

## Semantic Similarity

The generated answer and reference answer are embedded using:

```text
text-embedding-3-small
```

Their cosine similarity is calculated.

The resulting value indicates how semantically similar the generated answer is to the reference answer.

---

## LLM-as-a-Judge

The system also uses an LLM evaluator to assess the generated response.

Three dimensions are evaluated:

### Correctness

Whether the generated answer conveys the expected information.

### Relevance

Whether the generated answer directly addresses the question.

### Faithfulness

Whether the generated answer is supported by the retrieved context.

Each dimension is scored from:

```text
1 → 5
```

Where:

```text
5 = Completely correct / highly relevant / fully supported
4 = Mostly correct with minor issues
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect
```

The application also displays normalized percentage scores.

---

# Streamlit Interface

The application provides an interactive interface containing:

### Document Upload

Upload multiple PDFs simultaneously.

### Search Scope

Choose:

```text
All Documents
```

or a specific uploaded PDF.

### Question Answering

Ask natural-language questions about the uploaded documents.

### Retrieved Query

View the rewritten query used by the retrieval pipeline.

### Citations

View the source PDF and page number associated with retrieved information.

### Retrieved Evidence

Inspect the individual chunks retrieved from the vector database.

### Retrieved Context

Inspect the complete context provided to the LLM.

### RAG Pipeline Trace

Inspect the different stages of the RAG pipeline.

### Answer Evaluation

Provide a reference answer and evaluate:

- Semantic similarity
- Correctness
- Relevance
- Faithfulness

---

# Project Structure

```text
rag-based-pdf-question-answering/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── documents/
│       └── Rahul.pdf
│
├── src/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── loader.py
│   ├── query_rewriter.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   ├── retriever.py
│   └── vector_store.py
│
└── evaluation/
    ├── evaluate_answer_quality.py
    ├── evaluate_answers.py
    ├── evaluate_end_to_end.py
    ├── evaluate_faithfulness.py
    ├── evaluate_generation.py
    ├── evaluate_llm_judge.py
    ├── evaluate_reranking.py
    ├── evaluate_retrieval.py
    ├── generation_judge.py
    ├── generation_questions.json
    ├── llm_judge.py
    ├── metrics.py
    └── questions.json
```

---

# Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| UI | Streamlit |
| RAG Framework | LangChain |
| Embedding Model | OpenAI `text-embedding-3-small` |
| Vector Database | FAISS |
| Retrieval | FAISS similarity search |
| Reranking | Cross-Encoder |
| LLM | OpenAI API |
| PDF Processing | Python/LangChain PDF processing |
| Evaluation | Semantic Similarity + LLM-as-a-Judge |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/RahulRay2511MC10/rag-based-pdf-question-answering.git
```

Move into the project directory:

```bash
cd rag-based-pdf-question-answering
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not commit the `.env` file to GitHub.

---

# Running the Application

Start the Streamlit application:

```bash
python3 -m streamlit run app.py
```

The application will open in your browser.

---

# Example Workflow

```text
1. Upload Rahul.pdf
2. Upload another PDF
3. Click "Process PDFs"
4. Select "All Documents"
5. Enter a question
6. Query rewriting
7. FAISS retrieves candidate chunks
8. Cross-Encoder reranks candidates
9. LLM generates the answer
10. Sources and page citations are displayed
11. Inspect retrieved evidence
12. Provide a reference answer
13. Evaluate the generated answer
```

---

# Example

Suppose the following PDFs are uploaded:

```text
Rahul.pdf
Research_Paper.pdf
Resume.pdf
```

The user can select:

```text
Search in → All Documents
```

or:

```text
Search in → Research_Paper.pdf
```

A query such as:

```text
What methodology was used in the proposed system?
```

will retrieve relevant chunks, rerank them, generate an answer, and display the corresponding source and page citations.

---

# Key Design Decisions

### Chunking

```text
Chunk Size = 500
Chunk Overlap = 100
```

Chunk overlap helps preserve contextual continuity between neighboring chunks.

### Two-Stage Retrieval

The system uses:

```text
Stage 1 → FAISS semantic retrieval
Stage 2 → Cross-Encoder reranking
```

This separates efficient candidate retrieval from more precise relevance scoring.

### Metadata Preservation

Each chunk maintains document metadata, allowing the system to provide:

```text
Source PDF
Page Number
Chunk Information
```

This is particularly important for multi-document retrieval and citation.

---

# Future Improvements

Potential extensions include:

- OCR support for scanned PDFs
- Table-aware PDF extraction
- Image and diagram understanding
- Vision-language models for multimodal PDFs
- Hybrid BM25 + vector retrieval
- Advanced citation verification
- Retrieval quality dashboards
- Persistent vector databases
- Conversation memory across sessions
- Streaming LLM responses
- Authentication and multi-user document management

---

# Author

**Rahul Ray**

M.Tech Mathematics and Computing  
Indian Institute of Technology, Patna

GitHub:  
https://github.com/RahulRay2511MC10
```

