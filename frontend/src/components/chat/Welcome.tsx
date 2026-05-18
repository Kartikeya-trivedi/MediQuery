import { useEffect, useRef, useState } from "react";
import { Activity, FileText, Search, Stethoscope, Pill, HeartPulse, Info } from "lucide-react";
import { Composer } from "./Composer";

interface UploadedDoc {
  filename: string;
  chunks: number;
  dedup_removed?: number;
}

interface WelcomeProps {
  onSend: (text: string, webSearch?: boolean) => void;
  uploadedDocs?: UploadedDoc[];
  onFileUpload?: (doc: UploadedDoc) => void;
}

const SUGGESTIONS = [
  {
    icon: Stethoscope,
    label: "Discharge analysis",
    prompt: "Analyze the key clinical findings, diagnoses, and medication changes from this patient's discharge summary.",
  },
  {
    icon: Pill,
    label: "Drug interactions",
    prompt: "Review all medications in this patient's records and identify potential drug-drug interactions or contraindications.",
  },
  {
    icon: Activity,
    label: "Lab trends",
    prompt: "Summarize the trend of lab values (CBC, BMP, LFTs) during this patient's hospital stay and flag any critical values.",
  },
  {
    icon: Search,
    label: "Differential diagnosis",
    prompt: "Based on the documented symptoms, lab findings, and imaging results, what differential diagnoses should be considered?",
  },
];

const CLINICAL_GREETINGS = [
  "Ready for clinical review",
  "Let's analyze the records",
  "Upload patient data to begin",
  "Clinical intelligence, ready",
  "Evidence-based answers await",
  "Your clinical copilot is here",
  "What case are we reviewing",
  "Ready for chart review",
];

const getTimeGreeting = () => {
  const h = new Date().getHours();
  if (h < 5) return "Late shift clinical review";
  if (h < 12) return "Good morning, clinician";
  if (h < 18) return "Good afternoon, clinician";
  return "Evening clinical review";
};

const pickGreeting = () => {
  const pool = Math.random() < 0.4 ? [getTimeGreeting()] : CLINICAL_GREETINGS;
  return pool[Math.floor(Math.random() * pool.length)];
};

export const Welcome = ({ onSend, uploadedDocs, onFileUpload }: WelcomeProps) => {
  const [greeting] = useState(pickGreeting);
  const [shown, setShown] = useState("");
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    let i = 0;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      i += 1;
      setShown(greeting.slice(0, i));
      if (i < greeting.length) {
        setTimeout(tick, 30 + Math.random() * 45);
      }
    };
    const startId = setTimeout(tick, 250);
    return () => {
      cancelled = true;
      clearTimeout(startId);
    };
  }, [greeting]);

  const isStreaming = shown.length < greeting.length;

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pt-16 pb-8 animate-slide-up">
      <div className="mb-10 flex flex-col items-center text-center">
        {/* MediQuery Logo */}
        <div className="mb-6 relative">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/80 shadow-[var(--shadow-elevated)] animate-glow">
            <HeartPulse className="h-8 w-8 text-primary-foreground" strokeWidth={2} />
          </div>
          <div className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-[hsl(var(--clinical-safe))] ring-2 ring-background animate-pulse-soft" />
        </div>
        <h1 className="font-display text-[32px] sm:text-[40px] leading-tight font-semibold tracking-tight">
          <span className="bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">Medi</span>
          <span>Query</span>
        </h1>
        <p className="mt-1 text-sm font-medium text-primary/80 tracking-wider uppercase">
          Clinical Decision Support
        </p>
        <p className="mt-4 font-display text-xl sm:text-2xl font-medium text-foreground/90">
          {shown}
          {isStreaming && (
            <span className="ml-1 inline-block h-[0.85em] w-[2px] -mb-0.5 bg-primary animate-pulse align-middle" />
          )}
        </p>
        <p className="mt-3 text-sm text-muted-foreground max-w-md">
          Upload MIMIC-III discharge summaries or clinical documents to begin evidence-based analysis
        </p>
        
        <div className="mt-6 flex items-start gap-2 rounded-lg border border-primary/20 bg-primary/5 p-3 text-left shadow-sm max-w-md w-full">
          <Info className="h-4 w-4 text-primary shrink-0 mt-0.5" />
          <p className="text-xs text-foreground/80 leading-relaxed">
            <span className="font-semibold text-primary">Note:</span> The first query may take up to 2-3 minutes as serverless GPU containers boot from a cold start. Subsequent queries will be nearly instantaneous.
          </p>
        </div>
      </div>

      <Composer onSend={onSend} placeholder="Ask a clinical question…" autoFocus uploadedDocs={uploadedDocs} onFileUpload={onFileUpload} />

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            onClick={() => onSend(s.prompt)}
            className="group flex items-center gap-2 rounded-xl border border-border bg-surface-elevated px-4 py-2.5 text-[13px] text-foreground/80 hover:border-primary/40 hover:text-foreground hover:shadow-[var(--shadow-soft)] transition-all duration-200"
          >
            <s.icon className="h-3.5 w-3.5 text-primary" />
            {s.label}
          </button>
        ))}
      </div>

      {/* System capabilities */}
      <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-surface-elevated/50 p-4 text-center">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
            <Activity className="h-4 w-4 text-primary" />
          </div>
          <span className="text-xs font-semibold text-foreground/80">Hybrid Retrieval</span>
          <span className="text-[10px] text-muted-foreground">BM25 + Dense vectors with cross-encoder reranking</span>
        </div>
        <div className="flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-surface-elevated/50 p-4 text-center">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
            <HeartPulse className="h-4 w-4 text-primary" />
          </div>
          <span className="text-xs font-semibold text-foreground/80">NLI Verification</span>
          <span className="text-[10px] text-muted-foreground">Faithfulness checks to prevent clinical hallucinations</span>
        </div>
        <div className="flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-surface-elevated/50 p-4 text-center">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
            <Stethoscope className="h-4 w-4 text-primary" />
          </div>
          <span className="text-xs font-semibold text-foreground/80">Dual-Model Inference</span>
          <span className="text-[10px] text-muted-foreground">Gemma 4 26B + Llama 3.1 8B with cost-aware routing</span>
        </div>
      </div>
    </div>
  );
};
