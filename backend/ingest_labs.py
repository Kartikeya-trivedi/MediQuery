import os
import csv
import sys
import modal

app = modal.App("mediquery-ingest-labs")

vol = modal.Volume.from_name("ktgpt-rag-models")
MOUNT = "/models"

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
    .add_local_dir(".", remote_path="/app/backend")
)


@app.function(
    image=ingest_image,
    volumes={MOUNT: vol},
    gpu="T4",
    timeout=86400,
)
def process_lab_data(limit_admissions: int = 100):
    """
    Ingests the structured Kaggle MIMIC-III subset by converting
    admissions, patients, and lab events into synthetic clinical text notes.
    """
    sys.path.insert(0, "/app/backend")
    
    import pandas as pd
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from retriever import HybridRetriever
    from chunker import semantic_chunk, minhash_dedup
    
    files_needed = ["patient.csv", "admissions.csv", "d_labitems.csv", "labevents.csv"]
    for f in files_needed:
        if not os.path.exists(f"{MOUNT}/{f}"):
            print(f"❌ Error: {MOUNT}/{f} not found.")
            print(f"Please upload all 4 files via: modal volume put ktgpt-rag-models {f} /")
            return

    print("📊 Loading CSVs into Memory...")
    patients_df = pd.read_csv(f"{MOUNT}/patient.csv")
    admissions_df = pd.read_csv(f"{MOUNT}/admissions.csv")
    d_labitems_df = pd.read_csv(f"{MOUNT}/d_labitems.csv")
    labevents_df = pd.read_csv(f"{MOUNT}/labevents.csv")

    # Merge lab events with lab item definitions to get readable names
    print("🧬 Merging Lab Data...")
    labs_merged = labevents_df.merge(d_labitems_df, on="ITEMID", how="left")

    print("🚀 Loading Models & Qdrant...")
    embedder = SentenceTransformer(f"{MOUNT}/multilingual-e5-large")
    reranker = CrossEncoder(f"{MOUNT}/ms-marco-MiniLM-L-6-v2")
    retriever = HybridRetriever(
        embedder=embedder,
        reranker=reranker,
        qdrant_path=f"{MOUNT}/qdrant_data",
    )

    processed_count = 0
    total_chunks = 0

    print("⚙️ Generating Synthetic Clinical Notes...")
    # Iterate through each admission
    for _, admission in admissions_df.iterrows():
        subject_id = admission["SUBJECT_ID"]
        hadm_id = admission["HADM_ID"]
        
        # Get patient info
        patient = patients_df[patients_df["SUBJECT_ID"] == subject_id]
        gender = patient["GENDER"].iloc[0] if not patient.empty else "Unknown"
        
        # Get labs for this admission
        labs = labs_merged[labs_merged["HADM_ID"] == hadm_id]
        
        # Build a synthetic clinical note
        note_lines = [
            f"CLINICAL SUMMARY FOR PATIENT {subject_id} (GENDER: {gender})",
            f"ADMISSION ID: {hadm_id}",
            f"ADMIT TIME: {admission.get('ADMITTIME', 'Unknown')}",
            f"DIAGNOSIS AT ADMISSION: {admission.get('DIAGNOSIS', 'Unknown')}",
            f"ADMISSION TYPE: {admission.get('ADMISSION_TYPE', 'Unknown')}",
            f"DISCHARGE TIME: {admission.get('DISCHTIME', 'Unknown')}",
            "\nLABORATORY EVENTS & TRENDS:"
        ]
        
        if labs.empty:
            note_lines.append("No laboratory events recorded for this admission.")
        else:
            # Sort labs by time
            if "CHARTTIME" in labs.columns:
                labs = labs.sort_values(by="CHARTTIME")
                
            for _, lab in labs.iterrows():
                val = lab.get("VALUENUM", str(lab.get("VALUE", "")))
                uom = lab.get("VALUEUOM", "")
                flag = f" (FLAG: {lab['FLAG']})" if pd.notna(lab.get("FLAG")) else ""
                note_lines.append(
                    f"- {lab.get('CHARTTIME', 'Unknown Time')}: "
                    f"{lab.get('LABEL', 'Unknown Lab')} [{lab.get('FLUID', '')}] = "
                    f"{val} {uom}{flag}"
                )
                
        full_text = "\n".join(note_lines)
        doc_name = f"Patient_{subject_id}_Admit_{hadm_id}_Labs"
        
        # Semantic chunking
        chunks = semantic_chunk(full_text, embedder)
        chunks, removed = minhash_dedup(chunks)
        
        if chunks:
            retriever.ingest(doc_name, chunks)
            total_chunks += len(chunks)
            print(f"✅ Ingested {doc_name} ({len(chunks)} chunks)")
            
        processed_count += 1
        if limit_admissions > 0 and processed_count >= limit_admissions:
            break

    print(f"\n🎉 Complete! Processed {processed_count} admissions and ingested {total_chunks} chunks.")

@app.local_entrypoint()
def main(limit: int = 100):
    process_lab_data.remote(limit)
