import fitz  # PyMuPDF
import re
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the fast dense embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def parse_pdf_and_chunk(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"

    # Regex to find standard headers (e.g., "IS 269 : 1989" or "IS 458: 2003")
    standard_pattern = re.compile(r'(IS\s+\d+(?:\s*\(Part\s*\d+\))?\s*:\s*\d{4})')
    
    # Split the document by the Standard IDs
    splits = standard_pattern.split(full_text)
    
    corpus = []
    metadata = []
    
    # splits[0] is usually preamble/TOC. 
    # splits[1] is the first ID, splits[2] is its text, splits[3] is the next ID, etc.
    for i in range(1, len(splits)-1, 2):
        std_id = splits[i].strip()
        std_text = splits[i+1].strip()
        
        # Clean up text by removing excess whitespace
        std_text = re.sub(r'\s+', ' ', std_text)
        
        # PSEUDO-GRAPH: Find all other IS standards mentioned *inside* this standard's text
        references = list(set(standard_pattern.findall(std_text)))
        # Remove self-references
        references = [ref for ref in references if ref != std_id]
        
        # CHUNKING: Split the parent standard into smaller child paragraphs
        sentences = std_text.split(". ")
        chunk_size = 4
        
        for j in range(0, len(sentences), chunk_size):
            chunk_text = ". ".join(sentences[j:j+chunk_size])
            if len(chunk_text) < 50: # Skip tiny fragments
                continue
                
            corpus.append(chunk_text)
            metadata.append({
                "Standard_ID": std_id,
                "References": references
            })
            
    return corpus, metadata

def main():
    print("Parsing PDF and extracting Pseudo-Graph chunks...")
    corpus, metadata = parse_pdf_and_chunk("../dataset.pdf")
    
    print(f"Generated {len(corpus)} child chunks. Computing embeddings...")
    # Pre-compute embeddings to save massive latency during inference
    corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)
    
    # Save the knowledge base to disk
    print("Saving index to disk...")
    with open('../data/knowledge_base.pkl', 'wb') as f:
        pickle.dump({
            "corpus": corpus,
            "metadata": metadata,
            "embeddings": corpus_embeddings
        }, f)
        
    print("Ingestion complete. Ready for inference.")

if __name__ == "__main__":
    main()