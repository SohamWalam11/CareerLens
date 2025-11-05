# UX Copy Integration Guide

This document shows how to use the UX copy constants throughout the CareerLens frontend.

## Files

- **`UX_COPY.md`** — Human-readable reference with tone guide and all copy snippets
- **`frontend/src/lib/uxCopy.ts`** — TypeScript constants exported as `uxCopy` object
- **`frontend/src/components/Toast.tsx`** — Toast component with predefined message helpers
- **`frontend/src/components/Privacy.tsx`** — Privacy footer and onboarding prompt components

## Usage Examples

### 1. Onboarding Form Labels & Helpers

```tsx
import { uxCopy } from "../lib/uxCopy";

export function OnboardingStep1() {
  return (
    <label className="grid gap-2 text-sm">
      <span className="text-slate-300">{uxCopy.onboarding.step1.fullName.label}</span>
      <input placeholder="Your full name" />
      <p className="text-xs text-slate-500">{uxCopy.onboarding.step1.fullName.helper}</p>
    </label>
  );
}
```

### 2. Success/Error Toasts

```tsx
import { useToastMessages } from "../components/Toast";

export function MyComponent() {
  const toasts = useToastMessages();

  const handleSave = async () => {
    try {
      await api.saveProfile(data);
      toasts.profileSaved();  // "Profile saved! Generating your recommendations..."
    } catch (err) {
      toasts.profileSaveFailed();  // "Unable to save profile. Check your connection..."
    }
  };

  return <button onClick={handleSave}>Save</button>;
}
```

### 3. Recommendation Cards

```tsx
import { uxCopy } from "../lib/uxCopy";

export function RecommendationCard({ recommendation }) {
  return (
    <article>
      <h3>{recommendation.title}</h3>
      <p>{uxCopy.recommendations.cardHeader}</p>
      <div>
        <span className="badge">{uxCopy.recommendations.fitLabel(92)}</span>
      </div>
      <section>
        <h4>{uxCopy.recommendations.sections.keyRationale}</h4>
        {/* reasons here */}
      </section>
      <button>{uxCopy.recommendations.cta}</button>
    </article>
  );
}
```

### 4. Empty States

```tsx
import { uxCopy } from "../lib/uxCopy";

export function NoRecommendations() {
  const empty = uxCopy.emptyStates.noRecommendations;
  return (
    <div>
      <h2>{empty.heading}</h2>
      <p>{empty.body}</p>
      <button>{empty.cta}</button>
    </div>
  );
}
```

### 5. Privacy Notice

```tsx
import { PrivacyFooter, PrivacyOnboardingPrompt } from "../components/Privacy";

export function App() {
  return (
    <>
      {/* Your app */}
      <PrivacyFooter />
      {/* Or in onboarding */}
      <PrivacyOnboardingPrompt onDismiss={() => {}} />
    </>
  );
}
```

### 6. Navigation Tooltips

```tsx
import { uxCopy } from "../lib/uxCopy";

export function Navbar() {
  return (
    <nav>
      <a href="/recommendations" title={uxCopy.navigation.recommendations.tooltip}>
        {uxCopy.navigation.recommendations.label}
      </a>
      <a href="/trajectory" title={uxCopy.navigation.trajectory.tooltip}>
        {uxCopy.navigation.trajectory.label}
      </a>
    </nav>
  );
}
```

## Toast Provider Setup

Wrap your app root with the `ToastsProvider`:

```tsx
// main.tsx or App.tsx
import { ToastsProvider } from "./components/Toast";

function App() {
  return (
    <ToastsProvider>
      {/* Your app routes */}
    </ToastsProvider>
  );
}
```

Then use anywhere:

```tsx
import { useToasts } from "../components/Toast";

function SomeComponent() {
  const { showSuccess, showError } = useToasts();

  showSuccess("Custom success message");
  showError("Custom error message");
}
```

## Updating Copy

All user-facing text is centralized in `uxCopy.ts`. To update copy:

1. Edit the string in `uxCopy.ts`
2. It automatically updates across the app
3. No need to hunt through components

Example:

```tsx
// Before
export const uxCopy = {
  onboarding: {
    step1: {
      fullName: {
        helper: "We use this to personalize your career journey.",
      }
    }
  }
}

// After
export const uxCopy = {
  onboarding: {
    step1: {
      fullName: {
        helper: "Personalize your career journey.",  // Simplified
      }
    }
  }
}
```

## Tone Checklist

When writing or updating copy, ensure:

- ✅ **Encouraging:** "Unlock" not "Wait for"
- ✅ **Clear:** "Add skills you know" not "Configure capability matrix"
- ✅ **Action-oriented:** "View learning path" not "Path available"
- ✅ **Concise:** Max 120 characters (fits most screens)
- ✅ **Active voice:** "We matched you" not "You were matched"
- ✅ **Privacy-first:** Mention "anonymized" when relevant

## Testing Copy

### Manual

1. Run `npm run dev`
2. Navigate each page and verify copy matches `UX_COPY.md`
3. Test error states by mocking failed API calls

### Automated (Future)

```tsx
// Could add snapshot tests
describe("UX Copy", () => {
  test("toast messages are defined", () => {
    expect(uxCopy.toasts.errors.profileSaveFailed).toBeDefined();
  });
});
```

## Adding New Copy

1. Define in `UX_COPY.md` with tone guidelines
2. Add to `uxCopy.ts` object
3. Use in component: `import { uxCopy } from "../lib/uxCopy"`
4. Test on all devices (mobile especially; 120 char limit)

## Accessibility

All copy follows:

- Plain language (no jargon)
- Short sentences (easier to scan)
- Clear calls-to-action
- Adequate color contrast for text
- Descriptive labels for form fields

Toast component includes:

- `role="alert"` for screen readers
- `aria-live="polite"` for announcements
- Dismissible with keyboard (Tab to button)
- Manual dismiss button always available

---

## Reference

- **Tone Guide:** See `UX_COPY.md` "Copy Writing Tips"
- **Character Limits:** All snippets <120 chars
- **Toast Colors:**
  - Success: green (emerald)
  - Error: red
  - Info: blue
- **Privacy Mention:** Every page footer + onboarding
