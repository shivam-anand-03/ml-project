"use client";

import { Activity, BarChart3, Beaker, Brain, Cpu, Database, FileText, FlaskConical, LayoutDashboard, Settings, Sparkles, SlidersHorizontal, Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/analysis", label: "Experimental Analysis", icon: BarChart3 },
  { href: "/train", label: "Train Models", icon: Cpu },
  { href: "/evaluation", label: "Model Evaluation", icon: Activity },
  { href: "/predict", label: "AI Predictions", icon: Sparkles },
  { href: "/simulator", label: "What-If Simulator", icon: SlidersHorizontal },
  { href: "/explainability", label: "Explainability", icon: Brain },
  { href: "/validation", label: "Experimental Validation", icon: FlaskConical },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const health = useQuery({ queryKey: ["health"], queryFn: () => apiGet<{ status: string; database: boolean }>("/api/health"), refetchInterval: 15000, retry: false });
  const links = (
    <nav className="flex flex-col gap-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link key={href} href={href} className={clsx("nav-link", active && "active")} onClick={() => setOpen(false)}>
            <Icon size={17} /> {label}
          </Link>
        );
      })}
    </nav>
  );
  const status = health.isError ? "offline" : health.data?.status === "ok" ? "online" : "checking";
  return (
    <>
      <div className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--bg-elev)] px-4 py-3 lg:hidden">
        <Brand />
        <button onClick={() => setOpen((o) => !o)} className="text-[var(--muted)]">{open ? <X /> : <Menu />}</button>
      </div>
      {open && <div className="border-b border-[var(--border)] bg-[var(--bg-elev)] p-3 lg:hidden">{links}</div>}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--bg-elev)] p-4 lg:flex" style={{ minHeight: "100vh" }}>
        <Brand />
        <div className="mt-6">{links}</div>
        <div className="mt-auto pt-6 text-xs text-[var(--muted)]">
          <div className="flex items-center gap-2">
            <span className={clsx("inline-block h-2 w-2 rounded-full", status === "online" ? "bg-[var(--ok)]" : status === "offline" ? "bg-[var(--danger)]" : "bg-[var(--warn)]")} />
            Backend {status}
          </div>
          <div className="mt-2 leading-relaxed">All values labelled <span className="text-[var(--predicted)]">predicted</span> are model outputs; <span className="text-[var(--measured)]">measured</span> values come from your experiments.</div>
        </div>
      </aside>
    </>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-[var(--border-strong)] bg-[var(--accent-soft)] text-[var(--accent-2)]"><Beaker size={18} /></div>
      <div>
        <div className="text-sm font-bold leading-tight text-white">AI TriboPredict</div>
        <div className="text-[11px] text-[var(--muted)]">Data-enabled tribology</div>
      </div>
    </Link>
  );
}
