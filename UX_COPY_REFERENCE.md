# UX Copy Quick Reference

Cheat sheet for common messaging scenarios across CareerLens.

---

## Onboarding (4 Steps)

| Step | Label | Example Copy |
|------|-------|--------------|
| 1 | **Full name** | "We use this to personalize your career journey." |
| 1 | **Age** | "Helps us suggest roles matched to your career stage." |
| 1 | **Location** | "Optional. Enables location-specific insights." |
| 2 | **Education level** | "E.g., High School, Bachelor's, Master's, PhD." |
| 2 | **Primary focus** | "Your major, field of study, or specialization." |
| 2 | **Career vision** | "What role or impact do you want in 5 years?" |
| 3 | **Skills** | "Press Enter to add. E.g., Python, Project Management." |
| 3 | **Interests** | "Topics that excite you. E.g., AI Ethics, Analytics." |
| 4 | **Role** | "Job title or project name." |
| 4 | **Organization** | "Company, school project, or personal venture." |
| 4 | **Start year** | "Leave End year blank if still active." |
| 4 | **Highlights** | "Key achievements, impact, or skills gained." |

---

## Success Messages

```
"Profile saved! Generating your recommendations..."
"Your personalized recommendations are ready."
```

---

## Error Messages

```
"Unable to save profile. Check your connection and try again."
"We couldn't load recommendations right now. Reload or try again."
"Network error. Check your connection and try again."
"Please fill in all required fields before continuing."
```

---

## Empty States

| Scenario | Heading | Body | CTA |
|----------|---------|------|-----|
| **No profile** | "Start your journey" | "Complete your profile so we can find your perfect next role." | "Begin onboarding" |
| **No recommendations** | "Recommendations coming soon" | "Finish onboarding to unlock personalized career paths." | "Complete profile" |
| **No trajectory data** | "Trajectory not yet available" | "This role may be new or rarely chosen in your region." | "Explore other paths" |
| **No search results** | "No roles matched" | "Try searching with fewer filters or explore top recommendations." | "Clear filters" |

---

## Recommendation Cards

- **Header subtitle:** "Carefully matched to your skills and interests."
- **Fit score:** `Fit 92%`
- **Confidence badge:** `Confidence 87%`
- **Section labels:**
  - "Key rationale"
  - "Strength matches"
  - "Next skills to close the gap"
- **CTA:** "View learning path"

---

## Privacy

**Footer:**
```
Your privacy matters

We only store anonymized embeddings of your profile.
Your name, email, and personal data are never shared.
```

**Onboarding:**
```
We protect your data

Your profile is encrypted. Only anonymized skill embeddings 
are used for recommendations.
```

---

## Chat / AI Coach

- **Empty heading:** "Ask me anything"
- **Subtitle:** "I can explain role matches, skill gaps, or your next steps."
- **Starter prompt:** "Why is Data Scientist recommended for me?"
- **Input placeholder:** "Ask how to close the data storytelling gap..."

---

## Navigation

| Tab | Tooltip |
|-----|---------|
| **Recommendations** | "Your personalized career paths." |
| **Trajectory** | "Career progression map for your target role." |
| **AI Coach** | "Get personalized guidance on your career path." |

---

## Tone Rules (Apply to All Copy)

1. **Encouraging** — "Unlock" not "Wait"
2. **Clear** — Simple words, no jargon
3. **Action-oriented** — "View learning path" not "Path available"
4. **Concise** — Max **120 characters**
5. **Active voice** — "We matched you" not "You were matched"
6. **Privacy-first** — "Anonymized" when relevant

---

## Character Count Examples

✅ **Good (90 chars):**
```
We use this to personalize your career journey.
```

✅ **Good (115 chars):**
```
We only store anonymized embeddings of your profile.
Your name, email, and personal data are never shared.
```

❌ **Too long (140 chars):**
```
In order to provide you with the most personalized and 
tailored career recommendations based on your unique profile.
```

---

## Where Copy Lives

- **Human-readable:** `UX_COPY.md`
- **Code constants:** `frontend/src/lib/uxCopy.ts`
- **Components:** 
  - `Toast.tsx` (error/success)
  - `Privacy.tsx` (privacy notice)
  - `OnboardingWizard.tsx` (form labels)
  - `RecommendationCards.tsx` (card copy)

---

## How to Update Copy

1. **Edit in uxCopy.ts** (source of truth)
2. **Reference in component** (`import { uxCopy } from "../lib/uxCopy"`)
3. **No need to change multiple files** — changes propagate everywhere
4. **Update UX_COPY.md for documentation** (for non-devs)

Example:

```typescript
// frontend/src/lib/uxCopy.ts
export const uxCopy = {
  recommendations: {
    cardHeader: "Carefully matched to your skills and interests.",
    // ↑ Change here, applies everywhere
  }
};
```

---

## Testing Checklist

- [ ] All copy is encouraging (no negative language)
- [ ] No more than 120 characters per line
- [ ] Form helpers are visible on small screens
- [ ] Error messages are specific (not "error occurred")
- [ ] Privacy notice visible on every page
- [ ] Toasts disappear after 3–5 seconds (success faster)
- [ ] CTA buttons have clear action verbs ("View," "Explore," not "Click")

---

## Common Patterns

### Loading
```typescript
const [loading, setLoading] = useState(false);
return loading ? <p>{uxCopy.misc.loading}</p> : <Content />;
```

### Error
```typescript
const { showError } = useToasts();
catch (err) {
  showError(uxCopy.toasts.errors.networkError);
}
```

### Success
```typescript
const { showSuccess } = useToasts();
await api.save(data);
showSuccess(uxCopy.toasts.success.profileSaved);
```

### Empty State
```typescript
if (!data.length) {
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

---

## Resources

- **Full Guide:** `UX_COPY.md` (all snippets + tone guide)
- **Integration:** `UX_COPY_INTEGRATION.md` (how to use in code)
- **Code:** `frontend/src/lib/uxCopy.ts` (TypeScript source)
- **Components:** 
  - `Toast.tsx` (notifications)
  - `Privacy.tsx` (privacy footer)
