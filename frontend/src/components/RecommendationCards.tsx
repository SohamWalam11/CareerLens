import { useMemo } from "react";
import { SparklesIcon } from "@heroicons/react/24/outline";

import type { RecommendationBundle, SkillGap } from "../types/recommendations";

const clampScore = (fit: number) => Math.round(Math.min(1, Math.max(0, fit)) * 100);

const buildRingStyle = (percentage: number) => ({
  background: `conic-gradient(var(--tw-color-neon-blue) ${percentage}%, rgba(148, 163, 184, 0.15) ${percentage}% 100%)`
});

const parseMatchedSkills = (reasons: string[]): string[] => {
  const overlapLine = reasons.find((reason) => reason.toLowerCase().startsWith("skill overlap"));
  if (!overlapLine) return [];
  const match = overlapLine.match(/\(([^)]+)\)/);
  if (!match) return [];
  return match[1]
    .split(",")
    .map((skill) => skill.trim())
    .filter(Boolean)
    .slice(0, 4);
};

type Props = {
  bundles: RecommendationBundle[];
  onSelectLearningPlan?: (bundle: RecommendationBundle) => void;
};

const RecommendationCards = ({ bundles, onSelectLearningPlan }: Props) => {
  const cards = useMemo(() => bundles.slice(0, 5), [bundles]);

  if (!cards.length) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center text-sm text-slate-400">
        No recommendations yet. Complete onboarding to unlock personalized guidance.
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {cards.map((bundle: RecommendationBundle) => {
        const { role, explanation } = bundle;
        const fit = clampScore(role.fit_score);
        const matchedSkills = parseMatchedSkills(explanation.reasons);
        const gapSkills = explanation.gaps.slice(0, 3);

        return (
          <article
            key={role.title}
            className="flex flex-col justify-between rounded-2xl border border-slate-800 bg-slate-950/60 p-6 shadow transition hover:border-neon-pink/60 hover:shadow-neon"
          >
            <header className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-500">Recommended role</p>
                <h3 className="mt-1 text-xl font-semibold text-neon-blue">{role.title}</h3>
                {role.description && (
                  <p className="mt-2 text-sm text-slate-400">{role.description}</p>
                )}
              </div>
              <div className="relative h-16 w-16 rounded-full bg-slate-900" style={buildRingStyle(fit)}>
                <div className="absolute inset-2 flex flex-col items-center justify-center rounded-full bg-slate-950 text-xs font-semibold text-neon-blue">
                  <span>{fit}%</span>
                  <span className="text-[10px] text-slate-400">fit</span>
                </div>
              </div>
            </header>

            <dl className="mt-5 space-y-4 text-sm">
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Key rationale</dt>
                <dd className="mt-1 space-y-1 text-slate-300">
                  {explanation.reasons.slice(0, 2).map((reason: string) => (
                    <p key={reason} className="flex gap-2">
                      <SparklesIcon className="mt-0.5 h-4 w-4 text-neon-purple" />
                      <span>{reason}</span>
                    </p>
                  ))}
                </dd>
              </div>

              {matchedSkills.length > 0 && (
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500">Strength matches</dt>
                  <dd className="mt-2 flex flex-wrap gap-2">
                    {matchedSkills.map((skill: string) => (
                      <span
                        key={skill}
                        className="rounded-full border border-neon-blue/40 bg-neon-blue/10 px-3 py-1 text-xs font-semibold text-neon-blue"
                      >
                        {skill}
                      </span>
                    ))}
                  </dd>
                </div>
              )}

              {gapSkills.length > 0 && (
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500">Next skills</dt>
                  <dd className="mt-2 flex flex-wrap gap-2">
                    {gapSkills.map((gap: SkillGap) => (
                      <span
                        key={gap.skill}
                        className="rounded-full border border-neon-pink/40 bg-neon-pink/10 px-3 py-1 text-xs font-semibold text-neon-pink"
                        title={gap.suggested_action}
                      >
                        {gap.skill}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
            </dl>

            <footer className="mt-6 flex items-center justify-between gap-3">
              <div className="text-xs text-slate-500">Confidence {Math.round(explanation.confidence * 100)}%</div>
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-full bg-neon-purple px-4 py-2 text-xs font-semibold text-white transition hover:bg-neon-pink focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue"
                onClick={() => onSelectLearningPlan?.(bundle)}
              >
                View learning path
              </button>
            </footer>
          </article>
        );
      })}
    </div>
  );
};

export default RecommendationCards;
