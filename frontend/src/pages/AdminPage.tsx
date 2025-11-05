import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import apiClient from "../lib/api";
import type { AnalyticsSummary } from "../types/analytics";

const AdminPage = () => {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        setLoading(true);
        const response = await apiClient.get<AnalyticsSummary>("/analytics/summary");
        setSummary(response.data);
      } catch (err) {
        console.error("Failed to load analytics summary", err);
        setError("Unable to load analytics data. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchSummary().catch((err) => console.error("Unhandled analytics error", err));
  }, []);

  const topRoleData = useMemo(
    () =>
      (summary?.topRoles ?? []).map((item) => ({
        role: item.role,
        views: item.views,
        clicks: item.clicks
      })),
    [summary]
  );

  const averageScoreData = useMemo(
    () =>
      (summary?.averageScores ?? []).map((item) => ({
        event: item.eventType.replace(/_/g, " "),
        averageScore: Number((item.averageScore * 100).toFixed(1))
      })),
    [summary]
  );

  const feedbackHeatmapData = useMemo(() => {
    const baseMap = new Map<number, { rating: number; relevant: number; notRelevant: number }>();
    for (let rating = 1; rating <= 5; rating += 1) {
      baseMap.set(rating, { rating, relevant: 0, notRelevant: 0 });
    }

  (summary?.feedbackHeatmap ?? []).forEach((cell) => {
      const bucket = baseMap.get(cell.rating);
      if (!bucket) {
        return;
      }
      if (cell.relevant) {
        bucket.relevant += cell.count;
      } else {
        bucket.notRelevant += cell.count;
      }
    });

    return Array.from(baseMap.values()).sort((a, b) => b.rating - a.rating);
  }, [summary]);

  const totals = summary?.totals;

  return (
    <section className="space-y-8">
      <header>
        <h2 className="text-3xl font-semibold text-neon-blue">Admin Analytics</h2>
        <p className="mt-2 text-slate-400">
          Track recommendation engagement, feedback sentiment, and score quality in real time.
        </p>
      </header>

      {loading && <p className="text-slate-300">Loading analytics...</p>}
      {error && <p className="text-red-300">{error}</p>}

      {!loading && !error && summary && (
        <>
          <div className="grid gap-6 md:grid-cols-3">
            <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
              <p className="text-xs uppercase tracking-wide text-slate-600">Total Events</p>
              <p className="mt-2 text-3xl font-semibold text-slate-100">
                {totals?.totalEvents.toLocaleString() ?? "0"}
              </p>
              <p className="text-xs text-slate-500">Across all tracked actions</p>
            </article>
            {Object.entries(totals?.byType ?? {}).map(([eventType, count]) => (
              <article
                key={eventType}
                className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow"
              >
                <p className="text-xs uppercase tracking-wide text-slate-600">
                  {eventType.replace(/_/g, " ")}
                </p>
                <p className="mt-2 text-2xl font-semibold text-neon-blue">{count.toLocaleString()}</p>
                <p className="text-xs text-slate-500">Events recorded</p>
              </article>
            ))}
          </div>

          <section className="grid gap-6 lg:grid-cols-2">
            <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
              <h3 className="text-xl font-semibold text-neon-purple">Top Roles</h3>
              <p className="text-xs text-slate-500">View vs click engagement for recommended roles.</p>
              <div className="mt-6 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topRoleData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="role" tick={{ fill: "#cbd5f5", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#cbd5f5", fontSize: 12 }} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#0f172a", borderRadius: 8 }}
                      cursor={{ fill: "rgba(148, 163, 184, 0.1)" }}
                    />
                    <Legend />
                    <Bar dataKey="views" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="clicks" fill="#f472b6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </article>

            <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
              <h3 className="text-xl font-semibold text-neon-purple">Average Recommendation Scores</h3>
              <p className="text-xs text-slate-500">Mean fit scores for each interaction type.</p>
              <div className="mt-6 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={averageScoreData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="event" tick={{ fill: "#cbd5f5", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#cbd5f5", fontSize: 12 }} domain={[0, 100]} />
                    <Tooltip
                      formatter={(value) => [`${value}%`, "Average score"]}
                      contentStyle={{ backgroundColor: "#0f172a", borderRadius: 8 }}
                    />
                    <Line type="monotone" dataKey="averageScore" stroke="#a855f7" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </article>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
            <h3 className="text-xl font-semibold text-neon-purple">Feedback Heatmap</h3>
            <p className="text-xs text-slate-500">Stacked counts of ratings by perceived relevance.</p>
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={feedbackHeatmapData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="rating" tick={{ fill: "#cbd5f5", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#cbd5f5", fontSize: 12 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f172a", borderRadius: 8 }}
                    formatter={(value, name) => [value, name === "relevant" ? "Relevant" : "Not relevant"]}
                  />
                  <Legend wrapperStyle={{ color: "#cbd5f5" }} />
                  <Bar dataKey="relevant" stackId="heat" fill="#34d399" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="notRelevant" stackId="heat" fill="#f87171" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </>
      )}
    </section>
  );
};

export default AdminPage;
