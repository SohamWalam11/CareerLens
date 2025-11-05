import { useEffect, useState } from "react";

import apiClient from "../lib/api";
import { logAnalyticsEvent } from "../lib/analytics";
import type {
  RecommendationAPIResponse,
  RecommendationBundle,
  LearningResource
} from "../types/recommendations";

const mockRequestPayload = {
  name: "Alex Student",
  age: 21,
  education_level: "Undergraduate",
  interests: ["Data", "Design"],
  skills: ["Python", "SQL"],
  goals: ["Become Product Analyst"]
};

const RecommendationsPage = () => {
  const [data, setData] = useState<RecommendationAPIResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.post<RecommendationAPIResponse>(
          "/recommendations/generate",
          mockRequestPayload
        );
        setData(response.data);
      } catch (err) {
        console.error(err);
        setError("Unable to retrieve recommendations. Try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchData().catch((err) => console.error("Unhandled error", err));
  }, []);

  return (
    <section className="space-y-8">
      <header className="flex flex-col gap-2">
        <h2 className="text-3xl font-semibold text-neon-blue">Recommendations</h2>
        <p className="text-slate-400">
          Carefully curated career paths, skill gap analysis, and learning roadmap tailored to
          your latest profile.
        </p>
      </header>

      {loading && <p className="text-slate-300">Loading recommendations...</p>}

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-red-100">
          {error}
        </div>
      )}

      {data && (
        <div className="grid gap-6">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
            <h3 className="text-xl font-semibold text-neon-purple">Top Career Paths</h3>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {data.recommendations.map((item: RecommendationBundle) => (
                <article
                  key={item.role.title}
                  className="rounded-lg border border-slate-800 bg-slate-950/60 p-5 transition hover:border-neon-pink/60"
                >
                  <div className="flex items-center justify-between">
                    <h4 className="text-lg font-semibold text-neon-blue">{item.role.title}</h4>
                    <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-neon-pink">
                      Fit {Math.round(item.role.fit_score * 100)}%
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">{item.role.description}</p>
                  <div className="mt-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Trajectory</p>
                    <p className="mt-1 text-sm text-slate-300">{item.role.trajectory?.join(" → ")}</p>
                  </div>
                  <div className="mt-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Next Steps</p>
                    <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-300">
                      {item.role.next_steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="mt-6 flex justify-end">
                    <button
                      type="button"
                      onClick={() =>
                        logAnalyticsEvent({
                          eventType: "rec_clicked",
                          role: item.role.title,
                          score: item.role.fit_score,
                          context: { action: "view_learning_plan" }
                        })
                      }
                      className="rounded-md bg-neon-blue px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-neon-pink hover:text-white"
                    >
                      View learning plan
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="grid gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow">
              <h3 className="text-xl font-semibold text-neon-purple">Skill Gaps</h3>
              <ul className="mt-4 space-y-3">
                {data.recommendations.flatMap((r) => r.explanation.gaps).map((gap) => (
                <li
                    key={gap.skill}
                    className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-3"
                  >
                    <div>
                      <p className="text-sm font-semibold text-slate-200">{gap.skill}</p>
                      <p className="text-xs text-slate-500">{gap.reason}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            {/* Learning plan not currently provided by API - reserved for future enrichments */}
          </section>
        </div>
      )}
    </section>
  );
};

export default RecommendationsPage;
