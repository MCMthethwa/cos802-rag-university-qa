# cos802-rag-university-qa
# Evaluating Retrieval-Augmented Generation for Knowledge-Grounded Question Answering Using Public University Documents

## Overview
This project investigates whether Retrieval-Augmented Generation (RAG) can improve the accuracy, relevance, and grounding of answers generated from university documentation compared to a standalone language model.

A knowledge base was constructed from publicly available documents from the University of Pretoria (UP) and the University of KwaZulu-Natal (UKZN). The system uses semantic retrieval and a Large Language Model (LLM) to answer institution-specific questions regarding regulations, policies, programme requirements, and module information.

---

## Research Question
> Can Retrieval-Augmented Generation improve the accuracy and relevance of answers generated from university documents compared to a standalone language model?

---

## Objectives
* Develop a Retrieval-Augmented Generation (RAG) question-answering system.
* Construct a searchable university knowledge base from academic documents.
* Compare a standalone language model against a RAG-enhanced model.
* Evaluate retrieval effectiveness and answer quality.
* Investigate the impact of retrieval on reducing incorrect or unsupported responses.

---

## Dataset
The knowledge base was constructed using publicly available university documents.

### University of Pretoria (UP)
* General Academic Regulations and Student Rules (2026)
* EBIT Faculty Rules (2026)
* Programme Yearbook
* COS802 Study Guide
* COS802 Project Description
* Policy on Examinations and Related Matters
* UP Information Guide
* Code of Conduct for Students

### University of KwaZulu-Natal (UKZN)
* Student Rulebook (2026)

### Dataset Statistics

| Metric | Value |
| :--- | :--- |
| Documents Indexed | 9 |
| Searchable Chunks | 1,062 |
| Embedding Dimension | 384 |

---

## Methodology

### Offline Indexing Phase
1. PDF Extraction
2. Text Preprocessing
3. Document Chunking
4. Embedding Generation using Sentence Transformers
5. Vector Storage using FAISS

### Online Retrieval and Generation Phase
1. User Question
2. Query Embedding
3. Semantic Similarity Search
4. Top-K Chunk Retrieval
5. Answer Generation using FLAN-T5
6. Source Attribution

---

## Technology Stack

* **Programming Language:** Python
* **Libraries:**
  * PyPDF
  * Sentence Transformers
  * FAISS
  * Transformers
  * LangChain Text Splitters
  * Pandas
  * NumPy
* **Models:**
  * **Embedding Model:** `all-MiniLM-L6-v2`
  * **Language Model:** `google/flan-t5-base`

---

## Repository Structure
```text
cos802-rag-university-qa/
│
├── data/
│   ├── raw/
│   └── evaluation/
│
├── docs/
│   ├── proposal/
│   ├── literature-review/
│   ├── presentation/
│   └── report/
│
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_document_chunking.ipynb
│   ├── 03_embedding_generation.ipynb
│   ├── 04_vector_store_creation.ipynb
│   ├── 05_rag_pipeline.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_baseline_vs_rag_evaluation.ipynb
│
├── src/
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
│
└── README.md
```

---

## Experiments

### Retrieval Evaluation
The retrieval component was evaluated using ten university-related questions covering:
* Admission requirements
* Honours programmes
* Master's programmes
* Assessment regulations
* Student support services
* COS802-specific module information

#### Retrieval Results

| Metric | Result |
| :--- | :--- |
| Top-5 Retrieval Accuracy | 90% |

### Baseline vs RAG Evaluation
The RAG system was compared against a standalone language model using institution-specific questions.

| Metric | Result |
| :--- | :--- |
| Questions Evaluated | 10 |
| Average Baseline Score | 0.05 |
| Average RAG Score | 0.70 |
| RAG Wins | 8 |
| Baseline Wins | 0 |
| Ties | 2 |

### Example Result
* **Question:** What is the prerequisite for COS802?
* **Baseline Answer:** `a fusion reaction`
* **RAG Answer:** `MIT 801 [Intro to Machine Learning]`

*This illustrates how retrieval provides institution-specific knowledge that is unavailable to the standalone model.*

---

## Key Findings
* The retrieval system achieved 90% Top-5 Retrieval Accuracy.
* RAG consistently outperformed the standalone language model on institution-specific questions.
* Source attribution improved transparency and answer verification.
* Retrieval successfully identified relevant information across regulations, policies, study guides, and project documentation.
* Institution-specific questions benefited most from retrieval augmentation.

---

## Reproducibility
All source documents were downloaded and archived locally rather than referenced directly from university websites.

Maintaining a fixed document corpus ensures that experiments remain reproducible even if source documents are updated, relocated, or removed in the future.

---

## Future Work
* Improve retrieval ranking quality.
* Experiment with alternative embedding models.
* Evaluate larger language models.
* Expand the document collection to include additional universities.
* Introduce additional automated answer-quality metrics.
* Investigate cross-institution retrieval performance.

---

## Author
**Makhosazane Mthethwa**  
COS802 – Natural Language Processing  
University of Pretoria  
**Supervisor:** Sthembiso Mkhwanazi
