import { useMemo, useState } from "react";
import { CheckCircleIcon, PlusIcon, XMarkIcon } from "@heroicons/react/24/outline";
import type { ChangeEvent, FormEvent, KeyboardEvent } from "react";

import apiClient from "../lib/api";
import { getOrCreateUserId } from "../lib/user";
import { uxCopy } from "../lib/uxCopy";
import type { ExperienceEntry, OnboardingProfile } from "../types/profile";

const defaultProfile = (): OnboardingProfile => ({
  userId: getOrCreateUserId(),
  name: "Alex Student",
  age: 21,
  location: "Austin, TX",
  educationLevel: "Undergraduate",
  educationFocus: "Data Science",
  skills: ["Python", "SQL", "Storytelling"],
  interests: ["Product Analytics", "AI Ethics"],
  experience: [
    {
      id: crypto.randomUUID(),
      role: "Data Analyst Intern",
      organization: "Orbit Labs",
      startYear: "2023",
      endYear: "2024",
      highlights: "Built dashboards for marketing channel insights."
    }
  ],
  goals: "Become a product analytics lead"
});

const stepOrder = ["demographics", "education", "skills", "experience"] as const;
type StepId = (typeof stepOrder)[number];

const StepTitles: Record<StepId, string> = {
  demographics: "Profile basics",
  education: "Education",
  skills: "Skills & interests",
  experience: "Experience timeline"
};

const neonRing = (percentage: number) => ({
  background: `conic-gradient(var(--tw-color-neon-purple) ${percentage}%, rgba(148, 163, 184, 0.2) ${percentage}% 100%)`
});

type Props = {
  onProfileSaved?: (profile: OnboardingProfile) => void;
};

const OnboardingWizard = ({ onProfileSaved }: Props) => {
  const [profile, setProfile] = useState<OnboardingProfile>(() => defaultProfile());
  const [step, setStep] = useState<StepId>("demographics");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const stepIndex = stepOrder.indexOf(step);
  const completion = useMemo(() => Math.round(((stepIndex + 1) / stepOrder.length) * 100), [stepIndex]);

  const goToNext = () => {
    setStatus(null);
    setStep((prev: StepId) => {
      const idx = stepOrder.indexOf(prev);
      return stepOrder[Math.min(stepOrder.length - 1, idx + 1)];
    });
  };

  const goToPrevious = () => {
    setStatus(null);
    setStep((prev: StepId) => {
      const idx = stepOrder.indexOf(prev);
      return stepOrder[Math.max(0, idx - 1)];
    });
  };

  const handleTagKeyDown = (
    event: KeyboardEvent<HTMLInputElement>,
    listKey: "skills" | "interests"
  ) => {
    if (event.key !== "Enter" && event.key !== ",") return;
    event.preventDefault();
    const input = event.currentTarget;
    const value = input.value.trim();
    if (!value) return;
    setProfile((prev: OnboardingProfile) => ({
      ...prev,
      [listKey]: prev[listKey].includes(value) ? prev[listKey] : [...prev[listKey], value]
    }));
    input.value = "";
  };

  const removeTag = (value: string, listKey: "skills" | "interests") => {
    setProfile((prev: OnboardingProfile) => ({
      ...prev,
      [listKey]: prev[listKey].filter((item: string) => item !== value)
    }));
  };

  const addExperience = () => {
    const entry: ExperienceEntry = {
      id: crypto.randomUUID(),
      role: "",
      organization: "",
      startYear: "",
      endYear: "",
      highlights: ""
    };
    setProfile((prev: OnboardingProfile) => ({ ...prev, experience: [...prev.experience, entry] }));
  };

  const updateExperience = (id: string, key: keyof ExperienceEntry, value: string) => {
    setProfile((prev: OnboardingProfile) => ({
      ...prev,
      experience: prev.experience.map((entry: ExperienceEntry) =>
        entry.id === id
          ? {
              ...entry,
              [key]: value
            }
          : entry
      )
    }));
  };

  const removeExperience = (id: string) => {
    setProfile((prev: OnboardingProfile) => ({
      ...prev,
      experience: prev.experience.filter((entry: ExperienceEntry) => entry.id !== id)
    }));
  };

  const resolveStepTitle = (value: StepId) => StepTitles[value];

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setStatus(null);
    try {
      const payload = {
        user_id: profile.userId,
        name: profile.name,
        age: profile.age ?? 0,
        education_level: profile.educationLevel,
        interests: profile.interests,
        skills: profile.skills,
        goals: profile.goals
          .split(/[,\n]/)
      .map((item: string) => item.trim())
          .filter(Boolean)
      };

      await apiClient.post("/profile", payload);
      setStatus({ type: "success", message: uxCopy.toasts.success.profileSaved });
      onProfileSaved?.(profile);
    } catch (error) {
      console.error("Failed to save profile", error);
      setStatus({ type: "error", message: uxCopy.toasts.errors.profileSaveFailed });
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <div className="flex items-center justify-between gap-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow">
        <div>
          <p className="text-sm uppercase tracking-wide text-slate-500">Step {stepIndex + 1}</p>
          <h3 className="text-2xl font-semibold text-neon-blue">{resolveStepTitle(step)}</h3>
          <p className="text-sm text-slate-400">Fill in a few details so we can tailor your path.</p>
        </div>
        <div className="relative h-20 w-20 rounded-full bg-slate-900/80" style={neonRing(completion)}>
          <div className="absolute inset-2 flex items-center justify-center rounded-full bg-slate-950">
            <span className="text-lg font-semibold text-neon-purple">{completion}%</span>
          </div>
        </div>
      </div>

      <nav className="flex flex-wrap items-center gap-3 text-sm">
        {stepOrder.map((item, idx) => {
          const isActive = item === step;
          const isComplete = idx < stepIndex;
          return (
            <button
              key={item}
              type="button"
              onClick={() => setStep(item)}
              className={`flex items-center gap-2 rounded-full border px-4 py-2 transition focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 ${
                isActive
                  ? "border-neon-purple bg-neon-purple/20 text-neon-purple"
                  : isComplete
                  ? "border-neon-blue bg-neon-blue/10 text-neon-blue"
                  : "border-slate-700 bg-slate-900 text-slate-400 hover:border-neon-blue/40"
              }`}
            >
              {isComplete ? <CheckCircleIcon className="h-4 w-4" /> : <span className="h-2 w-2 rounded-full bg-current" />}
              {resolveStepTitle(item)}
            </button>
          );
        })}
      </nav>

      <section className="space-y-6 rounded-2xl border border-slate-800 bg-slate-950/60 p-8 shadow">
        {step === "demographics" && (
          <div className="grid gap-6 md:grid-cols-2">
            <label className="grid gap-2 text-sm">
              <span className="text-slate-300">{uxCopy.onboarding.step1.fullName.label}</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                value={profile.name}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setProfile({ ...profile, name: event.target.value })}
                required
                aria-label={uxCopy.onboarding.step1.fullName.label}
                placeholder={uxCopy.onboarding.step1.fullName.label}
              />
              <p className="text-xs text-slate-500">{uxCopy.onboarding.step1.fullName.helper}</p>
            </label>
            <label className="grid gap-2 text-sm">
              <span className="text-slate-300">{uxCopy.onboarding.step1.age.label}</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                type="number"
                min={0}
                value={profile.age ?? ""}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setProfile({ ...profile, age: Number(event.target.value) })}
                aria-label={uxCopy.onboarding.step1.age.label}
              />
              <p className="text-xs text-slate-500">{uxCopy.onboarding.step1.age.helper}</p>
            </label>
            <label className="grid gap-2 text-sm md:col-span-2">
              <span className="text-slate-300">{uxCopy.onboarding.step1.location.label}</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                value={profile.location}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setProfile({ ...profile, location: event.target.value })}
                placeholder="City, Country"
                aria-label={uxCopy.onboarding.step1.location.label}
              />
              <p className="text-xs text-slate-500">{uxCopy.onboarding.step1.location.helper}</p>
            </label>
          </div>
        )}

        {step === "education" && (
          <div className="grid gap-6 md:grid-cols-2">
            <label className="grid gap-2 text-sm">
              <span className="text-slate-300">Education level</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                value={profile.educationLevel}
                onChange={(event) => setProfile({ ...profile, educationLevel: event.target.value })}
                required
                aria-label="Education level"
              />
            </label>
            <label className="grid gap-2 text-sm">
              <span className="text-slate-300">Primary focus</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                value={profile.educationFocus}
                onChange={(event) => setProfile({ ...profile, educationFocus: event.target.value })}
                aria-label="Education focus"
              />
            </label>
            <label className="grid gap-2 text-sm md:col-span-2">
              <span className="text-slate-300">Career vision</span>
              <textarea
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-100 focus:border-neon-blue focus:outline-none"
                rows={3}
                value={profile.goals}
                onChange={(event) => setProfile({ ...profile, goals: event.target.value })}
                placeholder="Lead product analytics and mentor emerging analysts"
                aria-label="Career goals"
              />
            </label>
          </div>
        )}

        {step === "skills" && (
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-3">
              <span className="text-sm font-medium text-slate-300">Core skills</span>
              <div className="flex flex-wrap gap-2">
                {profile.skills.map((skill) => (
                  <button
                    key={skill}
                    type="button"
                    onClick={() => removeTag(skill, "skills")}
                    className="group flex items-center gap-2 rounded-full border border-neon-blue/50 bg-neon-blue/20 px-3 py-1 text-xs font-semibold text-neon-blue hover:bg-neon-blue/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue"
                    aria-label={`Remove skill ${skill}`}
                  >
                    {skill}
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                ))}
              </div>
              <input
                type="text"
                placeholder="Add a skill and press Enter"
                onKeyDown={(event) => handleTagKeyDown(event, "skills")}
                className="w-full rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-100 focus:border-neon-blue focus:outline-none"
                aria-label="Add skill"
              />
            </div>
            <div className="space-y-3">
              <span className="text-sm font-medium text-slate-300">Interests</span>
              <div className="flex flex-wrap gap-2">
                {profile.interests.map((interest) => (
                  <button
                    key={interest}
                    type="button"
                    onClick={() => removeTag(interest, "interests")}
                    className="group flex items-center gap-2 rounded-full border border-neon-purple/50 bg-neon-purple/20 px-3 py-1 text-xs font-semibold text-neon-purple hover:bg-neon-purple/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-purple"
                    aria-label={`Remove interest ${interest}`}
                  >
                    {interest}
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                ))}
              </div>
              <input
                type="text"
                placeholder="Add an interest and press Enter"
                onKeyDown={(event) => handleTagKeyDown(event, "interests")}
                className="w-full rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-100 focus:border-neon-purple focus:outline-none"
                aria-label="Add interest"
              />
            </div>
          </div>
        )}

        {step === "experience" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-300">Timeline</span>
              <button
                type="button"
                onClick={addExperience}
                className="inline-flex items-center gap-2 rounded-full border border-neon-blue px-3 py-1 text-xs font-semibold text-neon-blue hover:bg-neon-blue/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue"
              >
                <PlusIcon className="h-4 w-4" /> Add role
              </button>
            </div>
            <div className="space-y-4">
              {profile.experience.map((entry) => (
                <div
                  key={entry.id}
                  className="grid gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm md:grid-cols-2"
                >
                  <label className="grid gap-1">
                    <span className="text-slate-400">Role</span>
                    <input
                      className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                      value={entry.role}
                      onChange={(event) => updateExperience(entry.id, "role", event.target.value)}
                    />
                  </label>
                  <label className="grid gap-1">
                    <span className="text-slate-400">Organization</span>
                    <input
                      className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                      value={entry.organization}
                      onChange={(event) => updateExperience(entry.id, "organization", event.target.value)}
                    />
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    <label className="grid gap-1">
                      <span className="text-slate-400">Start year</span>
                      <input
                        className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                        value={entry.startYear}
                        onChange={(event) => updateExperience(entry.id, "startYear", event.target.value)}
                      />
                    </label>
                    <label className="grid gap-1">
                      <span className="text-slate-400">End year</span>
                      <input
                        className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                        value={entry.endYear}
                        onChange={(event) => updateExperience(entry.id, "endYear", event.target.value)}
                      />
                    </label>
                  </div>
                  <label className="md:col-span-2 grid gap-1">
                    <span className="text-slate-400">Highlights</span>
                    <textarea
                      rows={2}
                      className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-neon-blue focus:outline-none"
                      value={entry.highlights}
                      onChange={(event) => updateExperience(entry.id, "highlights", event.target.value)}
                    />
                  </label>
                  <div className="md:col-span-2 flex justify-end">
                    <button
                      type="button"
                      onClick={() => removeExperience(entry.id)}
                      className="inline-flex items-center gap-2 rounded-full border border-red-500/70 px-3 py-1 text-xs font-semibold text-red-200 hover:bg-red-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                    >
                      <XMarkIcon className="h-4 w-4" /> Remove
                    </button>
                  </div>
                </div>
              ))}
              {profile.experience.length === 0 && (
                <p className="rounded-lg border border-dashed border-slate-700 bg-slate-900/40 p-6 text-sm text-slate-400">
                  Add roles, internships, or major projects to help us plan your next move.
                </p>
              )}
            </div>
          </div>
        )}
      </section>

      <footer className="flex items-center justify-between">
        <button
          type="button"
          onClick={goToPrevious}
          className="rounded-full border border-slate-700 px-6 py-2 text-sm font-semibold text-slate-300 transition hover:border-neon-blue hover:text-neon-blue disabled:opacity-50"
          disabled={stepIndex === 0 || saving}
        >
          Back
        </button>
        {stepIndex < stepOrder.length - 1 ? (
          <button
            type="button"
            onClick={goToNext}
            className="rounded-full bg-neon-purple px-6 py-2 text-sm font-semibold text-white shadow transition hover:bg-neon-pink focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue"
          >
            Continue
          </button>
        ) : (
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-full bg-neon-pink px-6 py-2 text-sm font-semibold text-slate-950 shadow transition hover:bg-neon-purple focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue"
            disabled={saving}
          >
            {saving ? "Saving..." : "Save profile"}
          </button>
        )}
      </footer>

      {status && (
        <div
          role="status"
          className={`rounded-lg border px-4 py-3 text-sm ${
            status.type === "success"
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
              : "border-red-500/40 bg-red-500/10 text-red-100"
          }`}
        >
          {status.message}
        </div>
      )}
    </form>
  );
};

export default OnboardingWizard;
