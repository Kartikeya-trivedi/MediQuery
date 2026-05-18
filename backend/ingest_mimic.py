import os
import csv
import sys
import modal

app = modal.App("mediquery-ingest")

# Mount the same volume used by the RAG server
vol = modal.Volume.from_name("mediquery-rag-models")
MOUNT = "/models"

# Use the same image dependencies as the RAG Server
ingest_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
        "rank_bm25",
        "datasketch",
        "numpy",
        "qdrant-client>=1.7.0",
        "pandas"
    )
    # Include the backend code so we can import chunker & retriever
    .add_local_dir(".", remote_path="/app/backend")
)


@app.function(
    image=ingest_image,
    volumes={MOUNT: vol},
    gpu="T4",
    timeout=86400,  # Allow up to 24 hours for massive ingestion
)
def process_mimic_csv(csv_filename: str = "NOTEEVENTS.csv", limit: int = 100):
    """
    Reads the MIMIC-III NOTEEVENTS.csv from the Modal volume,
    chunks the discharge summaries, and ingests them into Qdrant.
    
    Args:
        csv_filename: The name of the CSV file uploaded to the Modal volume.
        limit: The maximum number of discharge summaries to process (useful for testing).
               Set to 0 to process ALL notes (will take a long time).
    """
    sys.path.insert(0, "/app/backend")
    
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from retriever import HybridRetriever
    from chunker import semantic_chunk, minhash_dedup
    
    csv_path = f"{MOUNT}/{csv_filename}"
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found on the Modal volume.")
        print(f"Please run: modal volume put mediquery-rag-models {csv_filename} /")
        return

    print("🚀 Loading Embedding Model...")
    embedder_path = f"{MOUNT}/multilingual-e5-large"
    embedder = SentenceTransformer(embedder_path)
    
    print("🚀 Loading Reranker...")
    reranker_path = f"{MOUNT}/ms-marco-MiniLM-L-6-v2"
    reranker = CrossEncoder(reranker_path)

    print("🚀 Initializing Qdrant Retriever...")
    qdrant_path = f"{MOUNT}/qdrant_data"
    retriever = HybridRetriever(
        embedder=embedder,
        reranker=reranker,
        qdrant_path=qdrant_path,
    )

    print(f"📂 Opening {csv_path}...")
    
    processed_count = 0
    total_chunks = 0
    
    # Increase CSV field size limit for large medical notes
    csv.field_size_limit(sys.maxsize)
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # We only want Discharge Summaries for the RAG system
            category = row.get("CATEGORY", "")
            if category.lower() != "discharge summary":
                continue
                
            subject_id = row.get("SUBJECT_ID", "Unknown")
            hadm_id = row.get("HADM_ID", "Unknown")
            text = row.get("TEXT", "")
            
            if not text.strip():
                continue
                
            doc_name = f"Patient_{subject_id}_Admission_{hadm_id}"
            print(f"\n⚙️ Processing {doc_name} ({len(text)} chars)...")
            
            # Semantic chunking
            chunks = semantic_chunk(text, embedder)
            
            # Deduplication
            chunks, removed = minhash_dedup(chunks)
            print(f"   -> Created {len(chunks)} chunks (removed {removed} duplicates)")
            
            if chunks:
                # Ingest into Qdrant
                retriever.ingest(doc_name, chunks)
                total_chunks += len(chunks)
                
            processed_count += 1
            if limit > 0 and processed_count >= limit:
                print(f"\n🛑 Reached limit of {limit} summaries.")
                break

    print("\n✅ Ingestion Complete!")
    print(f"Total Discharge Summaries processed: {processed_count}")
    print(f"Total Chunks added to Qdrant: {total_chunks}")
    print("\nYou can now check the RAG server stats to verify the index.")


@app.local_entrypoint()
def main(csv_filename: str = "NOTEEVENTS.csv", limit: int = 100):
    """
    CLI entrypoint.
    Run via: modal run backend/ingest_mimic.py --csv-filename NOTEEVENTS.csv --limit 100
    """
    print(f"Starting cloud ingestion job for {csv_filename} (limit: {limit})...")
    process_mimic_csv.remote(csv_filename, limit)
