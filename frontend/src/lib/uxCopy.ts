/**
 * CareerLens UX Copy Constants
 *
 * Centralized, easy-to-maintain copy for all user-facing text.
 * All strings follow the tone guide: encouraging, clear, <120 chars.
 */

export const uxCopy = {
  // ============================================
  // ONBOARDING HELPER TEXT
  // ============================================
  onboarding: {
    step1: {
      heading: "Profile basics",
      fullName: {
        label: "Full name",
        helper: "We use this to personalize your career journey.",
      },
      age: {
        label: "Age",
        helper: "Helps us suggest roles matched to your career stage.",
      },
      location: {
        label: "Location",
        helper: "Optional. Enables location-specific insights.",
      },
    },
    step2: {
      heading: "Education",
      educationLevel: {
        label: "Education level",
        helper: "E.g., High School, Bachelor's, Master's, PhD.",
      },
      educationFocus: {
        label: "Primary focus",
        helper: "Your major, field of study, or specialization.",
      },
      goals: {
        label: "Career vision",
        helper: "What role or impact do you want in 5 years?",
      },
    },
    step3: {
      heading: "Skills & interests",
      skills: {
        label: "Core skills",
        helper: "Press Enter to add. E.g., Python, Project Management.",
      },
      interests: {
        label: "Interests",
        helper: "Topics that excite you. E.g., AI Ethics, Analytics.",
      },
    },
    step4: {
      heading: "Experience timeline",
      addButton: "+ Add role",
      role: {
        label: "Role",
        helper: "Job title or project name.",
      },
      organization: {
        label: "Organization",
        helper: "Company, school project, or personal venture.",
      },
      startYear: {
        label: "Start year",
        helper: "Leave End year blank if still active.",
      },
      endYear: {
        label: "End year",
        helper: "Leave blank if currently active.",
      },
      highlights: {
        label: "Highlights",
        helper: "Key achievements, impact, or skills gained.",
      },
      emptyState: "Add roles, internships, or projects to unlock personalized recommendations.",
      cta: "Save profile",
    },
  },

  // ============================================
  // RECOMMENDATION CARDS
  // ============================================
  recommendations: {
    cardHeader: "Carefully matched to your skills and interests.",
    fitLabel: (percentage: number) => `Fit ${percentage}%`,
    confidenceLabel: (percentage: number) => `Confidence ${percentage}%`,
    sections: {
      keyRationale: "Key rationale",
      strengthMatches: "Strength matches",
      nextSkills: "Next skills to close the gap",
      trajectory: "Trajectory",
      nextSteps: "Next steps",
    },
    cta: "View learning path",
    exploreSimilar: "Explore similar roles →",
  },

  // ============================================
  // EMPTY STATES
  // ============================================
  emptyStates: {
    noProfile: {
      heading: "Start your journey",
      body: "Complete your profile so we can find your perfect next role.",
      cta: "Begin onboarding",
    },
    noRecommendations: {
      heading: "Recommendations coming soon",
      body: "Finish onboarding to unlock personalized career paths.",
      cta: "Complete profile",
    },
    noTrajectory: {
      heading: "Trajectory not yet available",
      body: "This role may be new or rarely chosen in your region.",
      cta: "Explore other paths",
    },
    noSearchResults: {
      heading: "No roles matched",
      body: "Try searching with fewer filters or explore top recommendations.",
      cta: "Clear filters",
    },
  },

  // ============================================
  // ERROR & SUCCESS TOASTS
  // ============================================
  toasts: {
    errors: {
      profileSaveFailed:
        "Unable to save profile. Please check your connection and try again.",
      recommendationLoadFailed:
        "We couldn't load recommendations right now. Reload the page or try again in a moment.",
      networkError:
        "Network error. Check your connection and try again.",
      validationError: "Please fill in all required fields before continuing.",
    },
    success: {
      profileSaved: "Profile saved! Generating your recommendations...",
      recommendationsReady: "Your personalized recommendations are ready.",
    },
  },

  // ============================================
  // PRIVACY & DATA
  // ============================================
  privacy: {
    heading: "Your privacy matters",
    body: "We only store anonymized embeddings of your profile. Your name, email, and personal data are never shared.",
    linkText: "Privacy Policy",
    onboardingHeading: "We protect your data",
    onboardingBody:
      "Your profile is encrypted. Only anonymized skill embeddings are used for recommendations.",
    onboardingCta: "Got it",
  },

  // ============================================
  // CHAT / AI COACH
  // ============================================
  chat: {
    emptyHeading: "Ask me anything",
    emptySubtitle:
      "I can explain role matches, skill gaps, or your next steps.",
    starterPrompt: "Why is Data Scientist recommended for me?",
    inputPlaceholder: "Ask how to close the data storytelling gap...",
  },

  // ============================================
  // NAVIGATION
  // ============================================
  navigation: {
    recommendations: {
      label: "Recommendations",
      tooltip: "Your personalized career paths.",
    },
    trajectory: {
      label: "Trajectory",
      tooltip: "Career progression map for your target role.",
    },
    coach: {
      label: "AI Coach",
      tooltip: "Get personalized guidance on your career path.",
    },
    home: {
      label: "Home",
    },
  },

  // ============================================
  // MISC
  // ============================================
  misc: {
    loading: "Loading...",
    tryAgain: "Try again",
    cancel: "Cancel",
    close: "Close",
    confirm: "Confirm",
    delete: "Delete",
  },
} as const;

export type UXCopy = typeof uxCopy;
