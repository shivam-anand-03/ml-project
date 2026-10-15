"use client";

import { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ReferenceLine, Area, ComposedChart } from "recharts";
import { SERIES_COLORS } from "@/lib/format";

export const AXIS = { stroke: "#5f7494", fontSize: 11 } as const;

export function ChartBox({ title, subtitle, children, height = 300, legend }: { title?: ReactNode; subtitle?: ReactNode; children: ReactNode; height?: number; legend?: ReactNode }) {
  return (
    <div className="panel p-4">
      {(title || legend) && (
        <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <div className="text-sm font-semibold text-white">{title}</div>}
            {subtitle && <div className="text-xs text-[var(--muted)]">{subtitle}</div>}
          </div>
          {legend}
        </div>
      )}
      <div style={{ width: "100%", height }}>{children}</div>
    </div>
  );
}

/** Multi-series line chart over a shared numeric x. series: [{name, points:[{x,y}]}] */
export function MultiLine({ series, xLabel, yLabel, yDomain, dashed, bandSeries }: { series: { name: string; points: { x: number; y: number | null }[]; color?: string; dashed?: boolean }[]; xLabel?: string; yLabel?: string; yDomain?: [number | "auto", number | "auto"]; dashed?: boolean; bandSeries?: { name: string; points: { x: number; lo: number; hi: number }[]; color?: string }[] }) {
  // merge into one array keyed by x (Recharts prefers a flat dataset); x values may differ per series so we use separate data per Line
  return (
    <ResponsiveContainer>
      <ComposedChart margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tick={AXIS} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -4, fill: "#8aa0bd", fontSize: 11 } : undefined} />
        <YAxis tick={AXIS} domain={yDomain || ["auto", "auto"]} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#8aa0bd", fontSize: 11 } : undefined} tickFormatter={(v) => (typeof v === "number" ? v.toFixed(3).replace(/\.?0+$/, "") : v)} />
        <Tooltip formatter={(v: number) => (typeof v === "number" ? v.toFixed(4) : v)} labelFormatter={(l) => `${xLabel || "x"}: ${typeof l === "number" ? l.toFixed(2) : l}`} />
        <Legend />
        {bandSeries?.map((b, i) => (
          <Area key={"band" + b.name} data={b.points.map((p) => ({ x: p.x, band: [p.lo, p.hi] }))} dataKey="band" name={b.name} stroke="none" fill={b.color || SERIES_COLORS[i % SERIES_COLORS.length]} fillOpacity={0.15} isAnimationActive={false} legendType="none" />
        ))}
        {series.map((s, i) => (
          <Line key={s.name} data={s.points} dataKey="y" name={s.name} stroke={s.color || SERIES_COLORS[i % SERIES_COLORS.length]} dot={false} strokeWidth={2} strokeDasharray={s.dashed || dashed ? "5 4" : undefined} isAnimationActive={false} connectNulls />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function ScatterXY({ points, xLabel, yLabel, identity, color, groups }: { points: { x: number; y: number; group?: string | null }[]; xLabel?: string; yLabel?: string; identity?: boolean; color?: string; groups?: boolean }) {
  const byGroup: Record<string, { x: number; y: number }[]> = {};
  for (const p of points) {
    const g = groups ? String(p.group ?? "all") : "all";
    (byGroup[g] ||= []).push({ x: p.x, y: p.y });
  }
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
  return (
    <ResponsiveContainer>
      <ScatterChart margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" name={xLabel} tick={AXIS} domain={["auto", "auto"]} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -4, fill: "#8aa0bd", fontSize: 11 } : undefined} />
        <YAxis dataKey="y" type="number" name={yLabel} tick={AXIS} domain={["auto", "auto"]} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#8aa0bd", fontSize: 11 } : undefined} />
        <ZAxis range={[14, 14]} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number) => v.toFixed(4)} />
        {Object.keys(byGroup).length > 1 && <Legend />}
        {identity && Number.isFinite(lo) && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#8aa0bd" strokeDasharray="4 4" />}
        {Object.entries(byGroup).map(([g, pts], i) => (
          <Scatter key={g} name={g} data={pts} fill={color || SERIES_COLORS[i % SERIES_COLORS.length]} fillOpacity={0.7} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function Bars({ data, xKey, bars, yLabel, horizontal, formatter }: { data: Record<string, unknown>[]; xKey: string; bars: { key: string; name?: string; color?: string }[]; yLabel?: string; horizontal?: boolean; formatter?: (v: number) => string }) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 8, right: 16, left: horizontal ? 40 : 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        {horizontal ? (
          <>
            <XAxis type="number" tick={AXIS} />
            <YAxis dataKey={xKey} type="category" tick={AXIS} width={120} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} tick={AXIS} />
            <YAxis tick={AXIS} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#8aa0bd", fontSize: 11 } : undefined} />
          </>
        )}
        <Tooltip formatter={(v: number) => (formatter ? formatter(v) : typeof v === "number" ? v.toFixed(4) : v)} />
        {bars.length > 1 && <Legend />}
        {bars.map((b, i) => (
          <Bar key={b.key} dataKey={b.key} name={b.name || b.key} fill={b.color || SERIES_COLORS[i % SERIES_COLORS.length]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Histogram({ bins, color, xLabel }: { bins: { x0: number; x1: number; count: number }[]; color?: string; xLabel?: string }) {
  const data = bins.map((b) => ({ x: ((b.x0 + b.x1) / 2).toFixed(3), count: b.count }));
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={AXIS} interval={Math.max(0, Math.floor(bins.length / 8))} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -4, fill: "#8aa0bd", fontSize: 11 } : undefined} />
        <YAxis tick={AXIS} />
        <Tooltip />
        <Bar dataKey="count" fill={color || SERIES_COLORS[0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Heatmap({ labels, matrix }: { labels: string[]; matrix: (number | null)[][] }) {
  const n = labels.length;
  const color = (v: number | null) => {
    if (v === null || v === undefined) return "#0c1627";
    const t = Math.max(-1, Math.min(1, v));
    return t >= 0 ? `rgba(47,140,255,${0.15 + 0.85 * t})` : `rgba(255,92,122,${0.15 + 0.85 * -t})`;
  };
  return (
    <div className="overflow-auto">
      <table className="text-[11px]">
        <thead>
          <tr>
            <th />
            {labels.map((l) => <th key={l} className="px-1 pb-1 text-[var(--muted)]" style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", maxHeight: 110 }}>{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: n }).map((_, i) => (
            <tr key={i}>
              <td className="pr-2 text-right text-[var(--muted)] whitespace-nowrap">{labels[i]}</td>
              {Array.from({ length: n }).map((_, j) => (
                <td key={j} title={`${labels[i]} vs ${labels[j]}: ${matrix[i][j]?.toFixed(3) ?? "–"}`} className="text-center font-medium text-white" style={{ background: color(matrix[i][j]), width: 46, height: 34, border: "1px solid #050a12" }}>
                  {matrix[i][j] === null ? "" : matrix[i][j]!.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LegendDots({ items }: { items: { name: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-[var(--muted)]">
      {items.map((i) => (
        <span key={i.name} className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: i.color }} />{i.name}</span>
      ))}
    </div>
  );
}

export { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, ReferenceLine };
