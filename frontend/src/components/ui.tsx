"use client";

import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import clsx from "clsx";
import { ReactNode, useEffect } from "react";

export function Panel({ children, className, title, subtitle, actions, glow }: { children: ReactNode; className?: string; title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; glow?: boolean }) {
  return (
    <section className={clsx("panel p-5", glow && "panel-glow", className)}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-base font-semibold text-white">{title}</h3>}
            {subtitle && <p className="mt-1 text-sm text-[var(--muted)]">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <div className="label mb-2">{children}</div>;
}

export function Stat({ label, value, hint, accent, unit }: { label: string; value: ReactNode; hint?: ReactNode; accent?: boolean; unit?: string }) {
  return (
    <div className="panel p-4">
      <div className="label">{label}</div>
      <div className={clsx("mt-2 text-2xl font-bold tracking-tight", accent ? "text-[var(--accent-2)]" : "text-white")}>
        {value}
        {unit && <span className="ml-1 text-sm font-medium text-[var(--muted)]">{unit}</span>}
      </div>
      {hint && <div className="mt-1 text-xs text-[var(--muted)]">{hint}</div>}
    </div>
  );
}

export function Badge({ children, tone = "muted", className }: { children: ReactNode; tone?: "muted" | "accent" | "ok" | "warn" | "danger" | "measured" | "predicted"; className?: string }) {
  const styles: Record<string, string> = {
    muted: "border-[var(--border-strong)] text-[var(--muted)]",
    accent: "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-soft)]",
    ok: "border-[var(--ok)] text-[var(--ok)] bg-[rgba(42,209,141,0.1)]",
    warn: "border-[var(--warn)] text-[var(--warn)] bg-[rgba(245,185,66,0.1)]",
    danger: "border-[var(--danger)] text-[var(--danger)] bg-[rgba(255,92,122,0.1)]",
    measured: "border-[var(--measured)] text-[var(--measured)]",
    predicted: "border-[var(--predicted)] text-[var(--predicted)]",
  };
  return <span className={clsx("badge", styles[tone], className)}>{children}</span>;
}

export function Alert({ tone = "info", title, children, className }: { tone?: "info" | "warn" | "danger" | "ok"; title?: ReactNode; children?: ReactNode; className?: string }) {
  const map = {
    info: { icon: Info, cls: "border-[var(--border-strong)] bg-[rgba(47,140,255,0.06)] text-[var(--text)]" },
    warn: { icon: AlertTriangle, cls: "border-[var(--warn)] bg-[rgba(245,185,66,0.08)] text-[var(--text)]" },
    danger: { icon: XCircle, cls: "border-[var(--danger)] bg-[rgba(255,92,122,0.08)] text-[var(--text)]" },
    ok: { icon: CheckCircle2, cls: "border-[var(--ok)] bg-[rgba(42,209,141,0.08)] text-[var(--text)]" },
  }[tone];
  const Icon = map.icon;
  return (
    <div className={clsx("flex gap-3 rounded-xl border p-3 text-sm", map.cls, className)}>
      <Icon size={18} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">
        {title && <div className="font-semibold">{title}</div>}
        {children && <div className={clsx(title && "mt-1", "text-[13px] leading-relaxed text-[var(--muted)]")}>{children}</div>}
      </div>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
      <span className="spinner" /> {label || "Loading…"}
    </div>
  );
}

export function Loading({ label }: { label?: string }) {
  return (
    <div className="panel flex items-center justify-center p-10">
      <Spinner label={label} />
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="panel flex flex-col items-center justify-center gap-3 p-10 text-center">
      <div className="text-base font-semibold text-white">{title}</div>
      {children && <div className="max-w-xl text-sm text-[var(--muted)]">{children}</div>}
      {action}
    </div>
  );
}

export function ErrorBox({ error, title }: { error: unknown; title?: string }) {
  const msg = error instanceof Error ? error.message : String(error);
  return <Alert tone="danger" title={title || "Request failed"}>{msg}</Alert>;
}

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={clsx("block", className)}>
      <div className="mb-1.5 text-xs font-semibold text-[var(--muted)]">{label}</div>
      {children}
      {hint && <div className="mt-1 text-[11px] text-[var(--muted-2)]">{hint}</div>}
    </label>
  );
}

export function Button({ children, variant = "default", size, className, loading, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "danger"; size?: "sm"; loading?: boolean }) {
  return (
    <button className={clsx("btn", variant === "primary" && "btn-primary", variant === "danger" && "btn-danger", size === "sm" && "btn-sm", className)} disabled={loading || rest.disabled} {...rest}>
      {loading && <span className="spinner" style={{ width: 14, height: 14 }} />}
      {children}
    </button>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { key: T; label: string }[]; value: T; onChange: (k: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1 rounded-xl border border-[var(--border)] bg-[#08101c] p-1">
      {tabs.map((t) => (
        <button key={t.key} onClick={() => onChange(t.key)} className={clsx("rounded-lg px-3 py-1.5 text-sm font-medium", value === t.key ? "bg-[var(--accent-soft)] text-white" : "text-[var(--muted)] hover:text-white")}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Modal({ open, onClose, title, children, footer }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; footer?: ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="panel panel-glow w-full max-w-lg p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-semibold text-white">{title}</h3>
          <button onClick={onClose} className="text-[var(--muted)] hover:text-white"><X size={18} /></button>
        </div>
        <div className="text-sm text-[var(--muted)]">{children}</div>
        {footer && <div className="mt-5 flex justify-end gap-2">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmDialog({ open, onClose, onConfirm, title, children, confirmLabel = "Confirm", danger, loading }: { open: boolean; onClose: () => void; onConfirm: () => void; title: ReactNode; children?: ReactNode; confirmLabel?: string; danger?: boolean; loading?: boolean }) {
  return (
    <Modal open={open} onClose={onClose} title={title} footer={<><Button onClick={onClose}>Cancel</Button><Button variant={danger ? "danger" : "primary"} onClick={onConfirm} loading={loading}>{confirmLabel}</Button></>}>
      {children}
    </Modal>
  );
}

export function DataTable({ columns, rows, maxHeight = 420, render }: { columns: string[]; rows: Record<string, unknown>[]; maxHeight?: number; render?: (col: string, value: unknown, row: Record<string, unknown>) => ReactNode }) {
  return (
    <div className="overflow-auto rounded-xl border border-[var(--border)]" style={{ maxHeight }}>
      <table className="table">
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr><td colSpan={columns.length} className="py-6 text-center text-[var(--muted)]">No rows</td></tr>
          )}
          {rows.map((r, i) => (
            <tr key={i}>{columns.map((c) => <td key={c}>{render ? render(c, r[c], r) : cell(r[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function cell(v: unknown): ReactNode {
  if (v === null || v === undefined) return <span className="text-[var(--muted-2)]">–</span>;
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(4);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function KeyValue({ items }: { items: Array<[ReactNode, ReactNode]> }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
      {items.map(([k, v], i) => (
        <div key={i} className="flex justify-between gap-3 border-b border-[var(--border)]/60 py-1.5">
          <dt className="text-[var(--muted)]">{k}</dt>
          <dd className="text-right font-medium text-white">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-[var(--muted)]">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
