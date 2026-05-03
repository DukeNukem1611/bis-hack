# BIS Standards Recommendation Engine: "Pseudo-Graph" Hybrid RAG

This repository contains our submission for the **Accelerating MSE Compliance - Automating BIS Standard Discovery** Hackathon track. 

Our solution is a highly optimized Retrieval-Augmented Generation (RAG) pipeline engineered specifically to handle the domain-specific jargon of the Bureau of Indian Standards (BIS) while strictly adhering to the <5 seconds average latency requirement.

##  Architecture & Code Logic

To achieve high accuracy (Hit Rate @3 > 80%, MRR @5 > 0.7) without failing the strict 5-second latency threshold, this pipeline splits the workload into two distinct phases: **Offline Ingestion** and **High-Speed Inference**.

### 1. Offline Data Ingestion (`src/ingest.py`)
This script processes the raw `dataset.pdf` (BIS SP 21) into a search-ready knowledge base. 
*   **Parent-Child Chunking:** Standard fixed-size chunking destroys legal context. We use Regex to split the document exactly at the Standard IDs (e.g., `IS 269: 1989`). This preserves the complete hierarchical context of each standard.
*   **"Pseudo-Graph" Metadata:** Standards frequently reference one another. During parsing, we extract any cross-referenced standards and inject them as metadata tags alongside the primary Standard ID.
*   **Pre-computed Embeddings:** To guarantee low latency during evaluation, we compute the dense vector embeddings using `all-MiniLM-L6-v2` *offline* and save them to a local Pickle file (`knowledge_base.pkl`).

### 2. High-Speed Inference (`inference.py`)
This is the entry-point script evaluated by the judges. It is engineered for maximum speed and exact JSON schema compliance.
*   **Memory Loading:** It instantly loads the pre-computed embeddings and metadata from disk, bypassing the need to encode the entire corpus at runtime.
*   **Hybrid Retrieval:** 
    *   **BM25 (Sparse):** Captures exact string matches (e.g., "33 Grade" or "PUB-01").
    *   **Dense Search:** Captures the semantic intent of the query using the `all-MiniLM-L6-v2` model.
*   **Cross-Encoder Reranking:** The combined top chunks are passed through a lightweight reranker (`ms-marco-MiniLM-L-6-v2`) to perfectly sort the most relevant context to the top.
*   **Zero-Generation Output:** To ensure a 0% JSON parsing failure rate and keep latency under 1 second per query, the script extracts the winning `Standard_ID` directly from the metadata of the reranked chunks rather than relying on a slow, error-prone LLM generation step.

---

## 📂 Project Structure
```text
BIS_Hackathon/                 
│
├── data/                      # Auto-generated during ingestion
│   └── knowledge_base.pkl     # Pre-computed index and embeddings
│
├── src/                       
│   └── ingest.py              # Offline data parsing and embedding script
│
├── dataset.pdf                # Official BIS SP 21 dataset (MUST be placed here)
├── public_test_set.json       # Official test queries
├── eval_script.py             # Official evaluation script
├── inference.py               # Main execution script for judges
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

##  How to Run

### Step 1: Install Dependencies
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment to keep your packages organized.
```bash
pip install -r requirements.txt
```

### Step 2: Run the Ingestion Pipeline (One-Time Setup)
Before running the inference script, you must parse the PDF and build the knowledge base. Ensure `dataset.pdf` is located in your root directory.
```bash
cd src
python ingest.py
cd ..
```
*Note: This will take a moment depending on your CPU/GPU. It will create a `knowledge_base.pkl` file inside the `data/` folder.*

### Step 3: Run Inference (Evaluation Simulation)
Execute the pipeline against the test queries. This simulates the exact command the judges will use to grade your project.
```bash
python inference.py --input public_test_set.json --output team_results.json
```
*This will quickly process the queries and generate a perfectly formatted `team_results.json` file in the root directory.*

### Step 5: Launch the UI
To test the web application interface for Micro & Small Enterprises (MSEs):
```bash
streamlit run app.py
```
