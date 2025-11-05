export type ExperienceEntry = {
  id: string;
  role: string;
  organization: string;
  startYear: string;
  endYear: string;
  highlights: string;
};

export type OnboardingProfile = {
  userId: string;
  name: string;
  age: number | null;
  location: string;
  educationLevel: string;
  educationFocus: string;
  skills: string[];
  interests: string[];
  experience: ExperienceEntry[];
  goals: string;
};
