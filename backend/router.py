"""
MediQuery — Cost-Aware Clinical Query Router
===============================================
Routes clinical queries between two model tiers based on
medical complexity heuristics:

- Small (Llama 3.1 8B Instruct): Simple factual lookups, single-variable queries
- Big (Gemma 4 26B-A4B-it): Complex differential diagnosis, multi-system analysis

Also supports escalation: if the small model produces an unfaithful response,
the query is automatically re-routed to the big model for safety-critical accuracy.
"""

import re


class QueryRouter:
    """Cost-aware routing between small and big LLM tiers for clinical queries.

    Decision signals:
    ┌──────────────────────────────────────────────┬───────────┬───────────┐
    │ Signal                                       │ → Small   │ → Big     │
    ├──────────────────────────────────────────────┼───────────┼───────────┤
    │ Query word count                             │ < 30      │ ≥ 30      │
    │ Retrieval confidence (reranker)              │ > 0.7     │ ≤ 0.7     │
    │ Clinical complexity keywords present         │ No        │ Yes       │
    │ Number of context chunks used                │ ≤ 2       │ > 2       │
    │ Previous small model was unfaithful           │ —         │ Escalate  │
    │ Safety-critical query detected               │ —         │ Escalate  │
    └──────────────────────────────────────────────┴───────────┴───────────┘

    Cost estimation (approximate):
    - Llama 3.1 8B:  ~$0.002/query (A10G, ~1.5s latency)
    - Gemma 4 26B:   ~$0.008/query (A100, ~3.0s latency)
    """

    # Clinical complexity keywords → route to big model
    COMPLEX_KEYWORDS = {
        "differential diagnosis", "differential", "differentials",
        "compare", "contrast", "analyze", "analyse",
        "drug interaction", "drug interactions", "contraindication",
        "contraindications", "polypharmacy",
        "comorbidity", "comorbidities", "multimorbidity",
        "pathophysiology", "etiology", "prognosis",
        "treatment plan", "treatment protocol", "therapeutic",
        "summarize", "summarise", "comprehensive",
        "step by step", "clinical reasoning", "clinical correlation",
        "critically", "in detail", "elaborate",
        "mortality", "morbidity", "risk stratification",
        "sepsis", "multi-organ", "multiorgan",
        "surgical", "perioperative", "postoperative",
        "hemodynamic", "ventilator", "mechanical ventilation",
        "renal replacement", "dialysis",
        "tumor staging", "cancer staging", "metastatic",
    }

    # Simple factual patterns → safe for small model
    SIMPLE_PATTERNS = [
        r"^what is\b",
        r"^what are\b",
        r"^who is\b",
        r"^when did\b",
        r"^where is\b",
        r"^define\b",
        r"^how many\b",
        r"^what was the\b",
        r"^what were the\b",
        r"^list the\b",
        r"^is it true\b",
        r"^yes or no\b",
        r"^what lab\b",
        r"^what medication\b",
        r"^what dose\b",
    ]

    # Safety-critical patterns → ALWAYS escalate to big model
    SAFETY_PATTERNS = [
        r"should (?:I|we|the patient) (?:stop|discontinue|change|switch)",
        r"is (?:it|this) (?:safe|dangerous|toxic|lethal|fatal)",
        r"overdos",
        r"anaphyla",
        r"code blue",
        r"cardiac arrest",
        r"life.?threatening",
        r"urgent|emergent|emergency",
    ]

    # Model identifiers
    SMALL = "llama"   # Llama 3.1 8B Instruct
    BIG = "gemma"     # Gemma 4 26B-A4B-it

    # Cost tracking (per-query estimated USD)
    COST_PER_QUERY = {
        "llama": 0.002,
        "gemma": 0.008,
    }

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_small_query_words: int = 30,
        max_small_chunks: int = 2,
    ):
        """
        Args:
            confidence_threshold: Below this, route to big model
            max_small_query_words: Queries longer than this go to big model
            max_small_chunks: If more chunks needed, route to big model
        """
        self.confidence_threshold = confidence_threshold
        self.max_small_query_words = max_small_query_words
        self.max_small_chunks = max_small_chunks
        self._total_cost = 0.0
        self._query_count = {"llama": 0, "gemma": 0}

    def route(
        self,
        query: str,
        retrieval_score: float = 0.0,
        context_chunks: int = 0,
    ) -> str:
        """Determine which model tier to route the clinical query to.

        Args:
            query: Clinician's question
            retrieval_score: Best reranker score from retrieval
            context_chunks: Number of retrieved context chunks

        Returns:
            'llama' for small model, 'gemma' for big model
        """
        signals = {
            "query_length": len(query.split()),
            "retrieval_score": retrieval_score,
            "context_chunks": context_chunks,
            "has_complex_keywords": self._has_complex_keywords(query),
            "is_simple_pattern": self._is_simple_pattern(query),
            "is_safety_critical": self._is_safety_critical(query),
        }

        # Safety-critical queries ALWAYS go to big model
        if signals["is_safety_critical"]:
            model = self.BIG
            print(f"🚨 Safety-critical query → Gemma 4 26B (mandatory escalation)")
            self._track_cost(model)
            return model

        # Decision logic: any complexity signal → big model
        reasons = []

        if signals["query_length"] > self.max_small_query_words:
            reasons.append(f"long query ({signals['query_length']} words)")

        if signals["has_complex_keywords"]:
            reasons.append("clinical complexity keywords detected")

        if retrieval_score > 0 and retrieval_score < self.confidence_threshold:
            reasons.append(f"low retrieval confidence ({retrieval_score:.3f})")

        if context_chunks > self.max_small_chunks:
            reasons.append(f"multi-chunk context ({context_chunks} chunks)")

        if reasons:
            model = self.BIG
            print(f"🧠 Routing → Gemma 4 26B (reasons: {', '.join(reasons)})")
        else:
            model = self.SMALL
            reason = "simple pattern" if signals["is_simple_pattern"] else "default"
            print(f"⚡ Routing → Llama 3.1 8B ({reason})")

        self._track_cost(model)
        return model

    def should_escalate(self, faithful: bool) -> bool:
        """Check if we should escalate from small to big model.

        Called after small model generation + faithfulness check.

        Args:
            faithful: Whether the small model's response was faithful

        Returns:
            True if we should re-generate with the big model
        """
        if not faithful:
            print("⬆️ Escalating to Gemma 4 26B due to unfaithful response (clinical safety)")
            return True
        return False

    def _has_complex_keywords(self, query: str) -> bool:
        """Check if query contains clinical complexity-indicating keywords."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.COMPLEX_KEYWORDS)

    def _is_simple_pattern(self, query: str) -> bool:
        """Check if query matches simple factual patterns."""
        query_lower = query.lower().strip()
        return any(re.match(p, query_lower) for p in self.SIMPLE_PATTERNS)

    def _is_safety_critical(self, query: str) -> bool:
        """Check if query involves safety-critical clinical decisions."""
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in self.SAFETY_PATTERNS)

    def _track_cost(self, model: str):
        """Track cumulative cost estimates."""
        cost = self.COST_PER_QUERY.get(model, 0.0)
        self._total_cost += cost
        self._query_count[model] = self._query_count.get(model, 0) + 1

    @property
    def cost_stats(self) -> dict:
        """Return cost tracking statistics."""
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "queries_small": self._query_count.get("llama", 0),
            "queries_big": self._query_count.get("gemma", 0),
            "cost_per_query_avg": round(
                self._total_cost / max(sum(self._query_count.values()), 1), 4
            ),
        }
