import { uxCopy } from "../lib/uxCopy";

export function PrivacyFooter() {
  return (
    <footer className="border-t border-slate-800 bg-slate-950/60 px-6 py-8 text-center text-sm text-slate-400">
      <div className="mx-auto max-w-2xl space-y-3">
        <h3 className="font-semibold text-slate-200">{uxCopy.privacy.heading}</h3>
        <p className="leading-relaxed">{uxCopy.privacy.body}</p>
        <a
          href="/privacy"
          className="inline-block text-neon-blue transition hover:text-neon-pink"
        >
          {uxCopy.privacy.linkText} →
        </a>
      </div>
    </footer>
  );
}

export function PrivacyOnboardingPrompt({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="rounded-lg border border-neon-blue/30 bg-neon-blue/10 p-4 text-sm">
      <h4 className="font-semibold text-neon-blue">{uxCopy.privacy.onboardingHeading}</h4>
      <p className="mt-2 text-slate-300">{uxCopy.privacy.onboardingBody}</p>
      <button
        onClick={onDismiss}
        className="mt-4 rounded-full border border-neon-blue/50 bg-neon-blue/20 px-4 py-2 text-xs font-semibold text-neon-blue transition hover:bg-neon-blue/30"
      >
        {uxCopy.privacy.onboardingCta}
      </button>
    </div>
  );
}
