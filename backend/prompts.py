"""
MediQuery — Clinical RAG Prompt Templates
============================================
Chat prompt formatters for Llama 3.1 Instruct and Gemma 4 Instruct
in clinical decision support contexts.

Each model uses its own special-token schema for optimal instruction following.
Prompts are grounded in evidence-based clinical language to ensure
high-fidelity diagnostic and therapeutic reasoning.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  System Prompt (shared across models)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediQuery, a clinical decision support assistant designed for healthcare professionals.

You provide evidence-based medical analysis grounded ONLY in the clinical context provided to you (e.g., MIMIC-III discharge summaries, clinical notes, lab reports).

STRICT RULES:
1. Answer ONLY using information from the provided clinical context.
2. If the context does not contain sufficient information, explicitly state: "Insufficient clinical evidence in the provided records to address this query."
3. NEVER fabricate diagnoses, lab values, medications, or clinical findings.
4. Always cite specific evidence from the context (e.g., "Per the discharge summary...", "Lab values indicate...").
5. Flag potential drug interactions, contraindications, or critical findings with appropriate clinical urgency.
6. Use standard medical terminology with brief explanations when appropriate.
7. Clearly distinguish between established findings and clinical reasoning/differential diagnoses.
8. Do NOT provide definitive treatment plans — frame recommendations as "considerations for clinical review."

You support clinical workflows including:
- Discharge summary analysis and key finding extraction
- Medication reconciliation and interaction screening
- Lab value trend interpretation
- Differential diagnosis reasoning from documented findings
- Clinical timeline reconstruction from EHR data"""


# ─────────────────────────────────────────────────────────────────────────────
#  Llama 3.1 Instruct Format
# ─────────────────────────────────────────────────────────────────────────────
def build_llama_prompt(context: str, question: str) -> str:
    """Build a prompt in Llama 3.1 Instruct chat format.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
    """
    if context.strip():
        user_content = (
            f"Clinical Records:\n{context}\n\n"
            f"Clinical Query: {question}"
        )
    else:
        user_content = question

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Gemma 4 Instruct Format
# ─────────────────────────────────────────────────────────────────────────────
def build_gemma_prompt(context: str, question: str) -> str:
    """Build a prompt in Gemma 4 chat format.

    Format:
        <start_of_turn>user
        {system + user}<end_of_turn>
        <start_of_turn>model
    """
    if context.strip():
        user_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Clinical Records:\n{context}\n\n"
            f"Clinical Query: {question}"
        )
    else:
        user_content = f"{SYSTEM_PROMPT}\n\n{question}"

    return (
        f"<start_of_turn>user\n"
        f"{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt Router
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(model_name: str, context: str, question: str) -> str:
    """Build a prompt for the specified model.

    Args:
        model_name: 'llama' or 'gemma'
        context: Retrieved clinical context (may be empty)
        question: Clinician's question
    """
    if model_name == "llama":
        return build_llama_prompt(context, question)
    elif model_name == "gemma":
        return build_gemma_prompt(context, question)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'llama' or 'gemma'.")
