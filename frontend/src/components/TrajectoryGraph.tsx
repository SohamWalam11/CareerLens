import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import apiClient from "../lib/api";
import type { TrajectoryNeighbor, TrajectoryResponse } from "../types/trajectory";

type NodeDatum = d3.SimulationNodeDatum & {
  id: string;
  label: string;
  type: "source" | "target";
  detail?: TrajectoryNeighbor;
};

type LinkDatum = d3.SimulationLinkDatum<NodeDatum> & {
  source: string | NodeDatum;
  target: string | NodeDatum;
};

type Props = {
  role: string;
};

const TrajectoryGraph = ({ role }: Props) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [data, setData] = useState<TrajectoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.get<TrajectoryResponse>("/trajectory", { params: { role } });
        if (isMounted) {
          setData(response.data);
        }
      } catch (err) {
        console.error("Failed to load trajectory", err);
        if (isMounted) {
          setError("Unable to load trajectory. Try again later.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchData().catch((err) => console.error("Unhandled trajectory error", err));
    return () => {
      isMounted = false;
    };
  }, [role]);

  useEffect(() => {
    if (!data || !svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth || 600;
    const height = 420;

    const nodes: NodeDatum[] = [
      { id: data.role, label: data.role, type: "source" }
    ];

    const links: LinkDatum[] = [];

    data.neighbors.forEach((neighbor: TrajectoryNeighbor) => {
      nodes.push({ id: neighbor.role, label: neighbor.role, type: neighbor.direction === "inbound" ? "target" : "target", detail: neighbor });
      links.push({
        source: neighbor.direction === "inbound" ? neighbor.role : data.role,
        target: neighbor.direction === "inbound" ? data.role : neighbor.role
      });
    });

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const zoomLayer = svg.append("g");

    const link = zoomLayer
      .append("g")
      .attr("stroke", "rgba(148, 163, 184, 0.3)")
      .attr("stroke-width", 1.5)
      .selectAll<SVGLineElement, LinkDatum>("line")
      .data(links)
      .enter()
      .append("line");

    const node = zoomLayer
      .append("g")
      .selectAll<SVGCircleElement, NodeDatum>("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", (d: NodeDatum) =>
        d.detail ? `${d.label}. Success rate ${Math.round(d.detail.success_rate * 100)} percent.` : d.label
      );

    node
      .append("circle")
      .attr("r", (d: NodeDatum) => (d.type === "source" ? 40 : 28))
      .attr("fill", (d: NodeDatum) => (d.type === "source" ? "rgba(127, 90, 240, 0.25)" : "rgba(0, 198, 255, 0.18)"))
      .attr("stroke", (d: NodeDatum) => (d.type === "source" ? "var(--tw-color-neon-purple)" : "var(--tw-color-neon-blue)"))
      .attr("stroke-width", 2.5);

    node
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("fill", "#f8fafc")
      .attr("font-size", (d: NodeDatum) => (d.type === "source" ? 14 : 12))
      .text((d: NodeDatum) => d.label);

    node.append("title").text((d: NodeDatum) => {
      if (!d.detail) return d.label;
      const { success_rate, avg_time_months, observed_transitions, common_skills_added } = d.detail;
      return [
        d.label,
        `Success rate: ${Math.round(success_rate * 100)}%`,
        `Average time: ${avg_time_months} months`,
        `Transitions observed: ${observed_transitions}`,
        common_skills_added.length ? `Common skills added: ${common_skills_added.join(", ")}` : ""
      ]
        .filter(Boolean)
        .join("\n");
    });

    const simulation = d3
      .forceSimulation(nodes as unknown as d3.SimulationNodeDatum[])
      .force("charge", d3.forceManyBody().strength(-320))
      .force(
        "link",
        // cast links for runtime; d3 will attach node refs when simulation runs
        d3.forceLink(links as unknown as d3.SimulationLinkDatum<d3.SimulationNodeDatum>[])
          .id((d: any) => d.id)
          .distance(160)
      )
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) => (d.type === "source" ? 60 : 46)));

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => (d.source?.x ?? 0))
        .attr("y1", (d: any) => (d.source?.y ?? 0))
        .attr("x2", (d: any) => (d.target?.x ?? 0))
        .attr("y2", (d: any) => (d.target?.y ?? 0));

      node.attr("transform", (d: any) => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    });

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 2.5])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        zoomLayer.attr("transform", event.transform.toString());
      });

    svg.call(zoom as unknown as (selection: d3.Selection<SVGSVGElement, unknown, null, undefined>) => void);

    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div ref={containerRef} className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Trajectory map</p>
          <h3 className="text-xl font-semibold text-neon-blue">{role}</h3>
        </div>
        {data?.centrality !== undefined && data?.centrality !== null && (
          <span className="rounded-full border border-neon-purple/40 bg-neon-purple/10 px-3 py-1 text-xs font-semibold text-neon-purple">
            Centrality {Math.round(data.centrality * 100)}%
          </span>
        )}
      </div>

      {loading && <p className="text-sm text-slate-400">Loading trajectory...</p>}
      {error && <p className="text-sm text-red-200">{error}</p>}

      <svg ref={svgRef} role="presentation" aria-hidden={loading || !!error} className="h-[420px] w-full" />

      {data && !loading && !error && (
        <div className="grid gap-3 text-xs text-slate-400 md:grid-cols-2">
          {data.neighbors.map((neighbor: TrajectoryNeighbor) => (
            <dl
              key={neighbor.role}
              className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
            >
              <dt className="text-sm font-semibold text-neon-blue">{neighbor.role}</dt>
              <dd className="mt-1">Direction: {neighbor.direction}</dd>
              <dd>Success rate: {Math.round(neighbor.success_rate * 100)}%</dd>
              <dd>Average time: {neighbor.avg_time_months} months</dd>
              <dd>Transitions: {neighbor.observed_transitions}</dd>
              {neighbor.common_skills_added.length > 0 && (
                <dd className="mt-1 text-[11px] uppercase tracking-wide text-slate-500">
                  Skills: {neighbor.common_skills_added.join(", ")}
                </dd>
              )}
            </dl>
          ))}
        </div>
      )}
    </div>
  );
};

export default TrajectoryGraph;
