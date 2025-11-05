type Milestone = {
  title: string;
  description: string;
  completed: boolean;
};

type Metric = {
  label: string;
  value: string;
  trend: string;
};

const milestones: Milestone[] = [
  {
    title: "Junior Data Analyst",
    description: "Establish solid foundation in analytics tooling and stakeholder reporting.",
    completed: true
  },
  {
    title: "Mid-Level Product Analyst",
    description: "Lead product experimentation, roadmap insights, and cross-functional alignment.",
    completed: false
  },
  {
    title: "Senior Insights Strategist",
    description: "Own analytics vision and mentor analysts across the organization.",
    completed: false
  }
];

const metrics: Metric[] = [
  { label: "Skill Gap Closure", value: "45%", trend: "+5%" },
  { label: "Learning Hours", value: "120", trend: "+20" },
  { label: "Feedback Score", value: "4.6/5", trend: "+0.2" }
];

const ProgressPage = () => {
  return (
    <section className="space-y-8">
      <header className="flex flex-col gap-2">
        <h2 className="text-3xl font-semibold text-neon-blue">Progress Dashboard</h2>
        <p className="text-slate-400">
          Track advancement across career milestones, learning velocity, and feedback loops.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-3">
  {metrics.map((metric: Metric) => (
          <div
            key={metric.label}
            className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow"
          >
            <p className="text-xs uppercase tracking-wide text-slate-500">{metric.label}</p>
            <p className="mt-2 text-3xl font-semibold text-slate-100">{metric.value}</p>
            <p className="text-xs text-neon-pink">{metric.trend} from last period</p>
          </div>
        ))}
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 shadow">
        <h3 className="text-xl font-semibold text-neon-purple">Career Trajectory</h3>
        <div className="mt-6 space-y-6">
          {milestones.map((milestone: Milestone, idx: number) => (
            <div key={milestone.title} className="flex items-start space-x-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-neon-blue bg-slate-950 text-sm font-semibold">
                {idx + 1}
              </div>
              <div>
                <h4 className="text-lg font-semibold text-slate-100">{milestone.title}</h4>
                <p className="text-sm text-slate-400">{milestone.description}</p>
                <span
                  className={`mt-2 inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${
                    milestone.completed
                      ? "bg-neon-blue/20 text-neon-blue"
                      : "bg-slate-800 text-slate-300"
                  }`}
                >
                  {milestone.completed ? "Completed" : "Upcoming"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
};

export default ProgressPage;
