import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

import apiClient from "../lib/api";
import { logAnalyticsEvent } from "../lib/analytics";

const defaultProfile = {
  name: "Alex Student",
  age: 21,
  educationLevel: "Undergraduate",
  interests: "Data Science, Product Design",
  skills: "Python, SQL, Communication",
  goals: "Become a product analyst"
};

const OnboardingPage = () => {
  const [profile, setProfile] = useState(defaultProfile);

  const handleChange = (field: keyof typeof defaultProfile) =>
    (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const value = event.target.value;
      setProfile({
        ...profile,
        [field]: field === "age" ? Number(value) || 0 : value
      });
    };

  const [status, setStatus] = useState<"idle" | "saving" | "success" | "error">("idle");

  const parseList = (value: string): string[] =>
    value
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const userId = profile.name.trim().toLowerCase().replace(/\s+/g, "-") || "demo-user";
    const payload = {
      user_id: userId,
      name: profile.name,
      age: Number(profile.age) || 0,
      education_level: profile.educationLevel,
      interests: parseList(profile.interests),
      skills: parseList(profile.skills),
      goals: parseList(profile.goals)
    };

    try {
      setStatus("saving");
  await apiClient.post("/profile", payload);
  await logAnalyticsEvent({ eventType: "profile_completed", userId, context: { source: "onboarding_page" } });
      setStatus("success");
    } catch (error) {
      console.error("Failed to save profile", error);
      setStatus("error");
    }
  };

  return (
    <section className="space-y-8">
      <header>
        <h2 className="text-3xl font-semibold text-neon-blue">Onboarding</h2>
        <p className="mt-2 max-w-2xl text-slate-400">
          Provide your background so CareerLens can tailor personalized recommendations.
        </p>
      </header>
      <form
        onSubmit={handleSubmit}
        className="grid grid-cols-1 gap-6 rounded-xl border border-slate-800 bg-slate-900/60 p-8 shadow-lg"
      >
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="name">
            Full Name
          </label>
          <input
            id="name"
            value={profile.name}
            onChange={handleChange("name")}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="age">
            Age
          </label>
          <input
            id="age"
            type="number"
            value={profile.age}
            onChange={handleChange("age")}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="education">
            Education Level
          </label>
          <input
            id="education"
            value={profile.educationLevel}
            onChange={handleChange("educationLevel")}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="interests">
            Interests
          </label>
          <textarea
            id="interests"
            value={profile.interests}
            onChange={handleChange("interests")}
            rows={3}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="skills">
            Skills
          </label>
          <textarea
            id="skills"
            value={profile.skills}
            onChange={handleChange("skills")}
            rows={3}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="grid gap-2">
          <label className="text-sm font-medium text-slate-300" htmlFor="goals">
            Goals
          </label>
          <textarea
            id="goals"
            value={profile.goals}
            onChange={handleChange("goals")}
            rows={3}
            className="rounded-md border border-slate-700 bg-slate-950 px-4 py-2 focus:border-neon-blue focus:outline-none"
          />
        </div>
        <div className="flex items-center justify-between gap-4">
          {status === "success" && (
            <p className="text-sm text-neon-blue">Profile saved! We will tailor recommendations.</p>
          )}
          {status === "error" && (
            <p className="text-sm text-red-300">Could not save profile. Please try again.</p>
          )}
          <button
            type="submit"
            className="rounded-md bg-neon-purple px-6 py-2 text-sm font-semibold text-white shadow hover:bg-neon-pink"
            disabled={status === "saving"}
          >
            {status === "saving" ? "Saving..." : "Save Profile"}
          </button>
        </div>
      </form>
    </section>
  );
};

export default OnboardingPage;
