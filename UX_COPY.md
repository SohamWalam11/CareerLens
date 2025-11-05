# CareerLens UX Copy Guide

Tone: Encouraging, clear, actionable. Max 120 characters per snippet.

---

## Onboarding Helper Text

### Step 1: Profile Basics

**Label:** Full name
**Helper:** We use this to personalize your career journey.

**Label:** Age
**Helper:** Helps us suggest roles matched to your career stage.

**Label:** Location
**Helper:** Optional. Enables location-specific insights.

---

### Step 2: Education

**Label:** Education level
**Helper:** E.g., High School, Bachelor's, Master's, PhD.

**Label:** Primary focus
**Helper:** Your major, field of study, or specialization.

**Label:** Career vision
**Helper:** What role or impact do you want in 5 years?

---

### Step 3: Skills & Interests

**Label:** Core skills
**Helper:** Press Enter to add. E.g., Python, Project Management.

**Label:** Interests
**Helper:** Topics that excite you. E.g., AI Ethics, Analytics.

---

### Step 4: Experience

**Label:** Role
**Helper:** Job title or project name.

**Label:** Organization
**Helper:** Company, school project, or personal venture.

**Label:** Start year / End year
**Helper:** Leave End year blank if still active.

**Label:** Highlights
**Helper:** Key achievements, impact, or skills gained.

**Add role CTA:** + Add role
**Empty state (experience):** Add roles, internships, or projects to unlock personalized recommendations.

---

## Recommendation Cards

### Card Header

**Subtitle (fit score):**
Fit 92%

**Body intro:**
Carefully matched to your skills and interests.

### Card Sections

**Key rationale section label:**
Key rationale

**Strength matches section label:**
Strength matches

**Next skills section label:**
Next skills to close the gap.

**Card footer CTA:**
View learning path

### Card Metadata

**Confidence badge:**
Confidence 87%

**Trajectory label:**
Trajectory

**Next steps label:**
Next steps

---

## Empty States

### No Profile Yet

**Heading:**
Start your journey

**Body:**
Complete your profile so we can find your perfect next role.

**CTA:**
Begin onboarding

---

### No Recommendations Yet

**Heading:**
Recommendations coming soon

**Body:**
Finish onboarding to unlock personalized career paths.

**CTA:**
Complete profile

---

### No Trajectory Data

**Heading:**
Trajectory not yet available

**Body:**
This role may be new or rarely chosen in your region.

**CTA:**
Explore other paths

---

### No Search Results

**Heading:**
No roles matched

**Body:**
Try searching with fewer filters or explore top recommendations.

---

## Error States & Toasts

### Profile Save Error

**Toast:**
Unable to save profile. Please check your connection and try again.

**Dismissible:** Yes (auto-dismiss 5s)

---

### Recommendation Load Error

**Toast:**
We couldn't load recommendations right now. Reload the page or try again in a moment.

**Dismissible:** Yes (auto-dismiss 5s)

---

### Network Error (General)

**Toast:**
Network error. Check your connection and try again.

**Dismissible:** Yes (auto-dismiss 5s)

---

### Validation Error (form)

**Toast:**
Please fill in all required fields before continuing.

**Dismissible:** Yes (manual)

---

## Success Messages

### Profile Saved

**Toast:**
Profile saved! Generating your recommendations...

**Dismissible:** Auto-dismiss (3s)

---

### Recommendation Generated

**Toast:**
Your personalized recommendations are ready.

**Dismissible:** Auto-dismiss (3s)

---

## Privacy & Data Note

### Footer / Settings Modal

**Heading:**
Your privacy matters

**Body:**
We only store anonymized embeddings of your profile.
Your name, email, and personal data are never shared.
Learn more: [Privacy Policy]

**Max length:** ~180 characters

---

**Privacy Policy Link Text:**
Privacy Policy

---

### Onboarding Privacy Prompt

**Heading:**
We protect your data

**Body:**
Your profile is encrypted. Only anonymized skill embeddings are used for recommendations.

**CTA:**
Got it

---

## Chat / AI Coach

### Empty State

**Heading:**
Ask me anything

**Subtitle:**
I can explain role matches, skill gaps, or your next steps.

**Starter prompt:**
Why is Data Scientist recommended for me?

---

### Placeholder Text

**Input placeholder:**
Ask how to close the data storytelling gap...

---

## Navigation & Misc

### Tab: Recommendations

**Label:**
Recommendations

**Tooltip (if hovered):**
Your personalized career paths.

---

### Tab: Trajectory

**Label:**
Trajectory

**Tooltip:**
Career progression map for your target role.

---

### Tab: Chat

**Label:**
AI Coach

**Tooltip:**
Get personalized guidance on your career path.

---

### Breadcrumb (if used)

Example: Home > Recommendations > Data Scientist

**Separator:** /

---

## Copy Writing Tips

1. **Avoid jargon.** Use "career stage" not "seniority level."
2. **Be specific.** "Add skills you're confident with" not "Add skills."
3. **Encourage action.** "See how to close this gap" not "Gap exists."
4. **Honor privacy.** Mention "anonymized" and "never shared" upfront.
5. **Keep it short.** 120 chars max ensures mobile readability.
6. **Use active voice.** "We matched you to" not "You were matched to."

---

## Color & Tone Reference

- **Success:** Green text + checkmark (e.g., "Profile saved!")
- **Error:** Red text + alert icon (e.g., "Unable to save profile.")
- **Info:** Blue text (e.g., "Generating recommendations...")
- **Warning:** Yellow/amber (e.g., "This may take a moment.")

---

## Examples in Context

### Onboarding Form Field
```
Label: Full name
Helper: We use this to personalize your career journey.
Input placeholder: Your full name
```

### Recommendation Card
```
Title: Data Scientist
Fit: 92%
Subtitle: Carefully matched to your skills and interests.
Reason 1: Skill overlap: matched 3 of 4 core skills.
Strength match: Python, SQL
Next skill: Machine Learning (6 months to mastery)
CTA: View learning path
Footer: Confidence 87% | Explore similar roles →
```

### Empty State
```
Icon: 🎯
Heading: Recommendations coming soon
Body: Finish onboarding to unlock personalized career paths.
CTA: Complete profile
```

### Error Toast
```
Icon: ⚠️
Message: Unable to save profile. Check your connection and try again.
Dismiss: Auto (5s) or manual
```

### Privacy Footer
```
Heading: Your privacy matters
Body: We only store anonymized embeddings of your profile.
     Your name, email, and personal data are never shared.
Link: Privacy Policy →
```
