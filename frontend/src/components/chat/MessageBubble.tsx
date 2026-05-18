import ReactMarkdown from "react-markdown";
import {
  Copy, RefreshCw, ThumbsDown, ThumbsUp,
  Zap, Brain, AlertTriangle, Bookmark, Shield,
  ShieldCheck, ShieldAlert, DollarSign, HeartPulse,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/lib/chatTypes";
import { useState } from "react";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

// ── Model Tier Badge ─────────────────────────────────────────────────────────
const ModelBadge = ({ modelUsed }: { modelUsed: string }) => {
  const isSmall = modelUsed.includes("llama");
  return (
    <span
      title={isSmall ? "Llama 3.1 8B — Fast tier (~$0.002/query)" : "Gemma 4 26B — Deep tier (~$0.008/query)"}
      className={cn(
        "clinical-badge ring-1",
        isSmall
          ? "bg-emerald-500/10 text-emerald-500 ring-emerald-500/20"
          : "bg-violet-500/10 text-violet-500 ring-violet-500/20"
      )}
    >
      {isSmall ? <Zap className="h-2.5 w-2.5" /> : <Brain className="h-2.5 w-2.5" />}
      {isSmall ? "Llama 3.1 8B" : "Gemma 4 26B"}
    </span>
  );
};

// ── Confidence Bar ───────────────────────────────────────────────────────────
const ConfidenceBar = ({ score }: { score: number }) => {
  const pct = Math.round(score * 100);
  const color =
    pct >= 70 ? "bg-[hsl(var(--clinical-safe))]" :
    pct >= 40 ? "bg-[hsl(var(--clinical-warning))]" :
    "bg-[hsl(var(--clinical-danger))]";
  const label =
    pct >= 70 ? "High" : pct >= 40 ? "Moderate" : "Low";
  return (
    <span
      title={`Evidence confidence: ${pct}% (${label})`}
      className="inline-flex items-center gap-1.5"
    >
      <span className="h-1.5 w-16 rounded-full bg-foreground/10 overflow-hidden">
        <span
          className={cn("h-full block rounded-full transition-all duration-500", color)}
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="text-[10px] text-muted-foreground font-mono">{pct}%</span>
    </span>
  );
};

// ── Cache Badge ──────────────────────────────────────────────────────────────
const CacheBadge = () => (
  <span
    title="Semantic cache hit — response served from cached clinical query"
    className="clinical-badge ring-1 bg-sky-500/10 text-sky-500 ring-sky-500/20"
  >
    <Bookmark className="h-2.5 w-2.5" />
    Cached
  </span>
);

// ── Clinical Risk Badge ─────────────────────────────────────────────────────
const ClinicalRiskBadge = ({ risk }: { risk: string }) => {
  if (risk === "low") {
    return (
      <span
        title="Clinical risk: LOW — Response is well-grounded in evidence"
        className="clinical-badge-safe"
      >
        <ShieldCheck className="h-2.5 w-2.5" />
        Evidence-based
      </span>
    );
  }
  if (risk === "medium") {
    return (
      <span
        title="Clinical risk: MEDIUM — Some claims may not be fully grounded"
        className="clinical-badge-warning"
      >
        <Shield className="h-2.5 w-2.5" />
        Verify claims
      </span>
    );
  }
  return (
    <span
      title="Clinical risk: HIGH — Response may contain ungrounded medical claims"
      className="clinical-badge-danger"
    >
      <ShieldAlert className="h-2.5 w-2.5" />
      Review required
    </span>
  );
};

// ── Faithfulness Warning ─────────────────────────────────────────────────────
const FaithfulnessWarning = () => (
  <span
    title="NLI check: response may not be fully entailed by the clinical context"
    className="clinical-badge-warning"
  >
    <AlertTriangle className="h-2.5 w-2.5" />
    NLI: Verify
  </span>
);

// ── Cost Badge ───────────────────────────────────────────────────────────────
const CostBadge = ({ cost }: { cost: number }) => (
  <span
    title={`Estimated inference cost: $${cost.toFixed(4)}`}
    className="clinical-badge ring-1 bg-foreground/5 text-muted-foreground ring-foreground/10"
  >
    <DollarSign className="h-2.5 w-2.5" />
    ${cost.toFixed(3)}
  </span>
);

export const MessageBubble = ({ message, isStreaming }: MessageBubbleProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  // Determine if we have any metadata to show
  const hasMetadata =
    !isStreaming &&
    message.content &&
    !isUser &&
    (message.modelUsed || message.confidence !== undefined || message.cached ||
     message.faithful === false || message.clinicalRisk || message.costUsd);

  if (isUser) {
    return (
      <div className="group flex justify-end animate-fade-in">
        <div className="max-w-[85%] rounded-2xl rounded-tr-md bg-user-bubble px-4 py-3 text-[15px] leading-relaxed text-user-bubble-foreground shadow-[var(--shadow-soft)]">
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex gap-4 animate-fade-in">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/10 ring-1 ring-primary/20">
        <HeartPulse className="h-3.5 w-3.5 text-primary" strokeWidth={2.5} />
      </div>
      <div className="min-w-0 flex-1 pt-1">
        <div className="prose-clinical text-[15px]">
          <ReactMarkdown>{message.content || "\u200B"}</ReactMarkdown>
          {isStreaming && (
            <span className="inline-block h-4 w-[2px] translate-y-0.5 bg-primary/70 ml-0.5 animate-blink" />
          )}
        </div>

        {/* Clinical Metadata Row */}
        {hasMetadata && (
          <div className="mt-3 flex flex-wrap items-center gap-2 pb-1 border-t border-border/50 pt-3">
            {message.clinicalRisk && (
              <ClinicalRiskBadge risk={message.clinicalRisk} />
            )}
            {message.modelUsed && <ModelBadge modelUsed={message.modelUsed} />}
            {message.cached && <CacheBadge />}
            {!message.cached && message.confidence !== undefined && message.confidence > 0 && (
              <ConfidenceBar score={message.confidence} />
            )}
            {message.faithful === false && <FaithfulnessWarning />}
            {message.costUsd !== undefined && message.costUsd > 0 && (
              <CostBadge cost={message.costUsd} />
            )}
          </div>
        )}

        {/* Source attribution */}
        {!isStreaming && message.source && (
          <div className="mt-2 text-[11px] text-muted-foreground/70 font-mono">
            📄 Sources: {message.source}
          </div>
        )}

        {/* Action buttons */}
        {!isStreaming && message.content && (
          <div className="mt-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <IconBtn onClick={copy} label={copied ? "Copied" : "Copy"}>
              <Copy className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Retry">
              <RefreshCw className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Clinically accurate">
              <ThumbsUp className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Needs review">
              <ThumbsDown className="h-3.5 w-3.5" />
            </IconBtn>
          </div>
        )}
      </div>
    </div>
  );
};

const IconBtn = ({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick?: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    title={label}
    className={cn(
      "flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground",
      "hover:bg-muted hover:text-foreground transition-colors"
    )}
  >
    {children}
  </button>
);
