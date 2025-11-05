export type SkillGap = {
  skill: string;
  reason: string;
  suggested_action: string;
};

export type RecommendationExplanation = {
  reasons: string[];
  gaps: SkillGap[];
  confidence: number;
};

export type LearningResource = {
  title: string;
  provider?: string;
  url?: string;
  estimated_hours?: number;
};

export type CareerPath = {
  title: string;
  fit_score: number;
  description?: string;
  next_steps: string[];
  trajectory: string[];
};

export type RecommendationBundle = {
  role: CareerPath;
  explanation: RecommendationExplanation;
};

export type ProfileSnapshot = {
  name: string;
  age: number;
  education_level: string;
  interests: string[];
  skills: string[];
  goals: string[];
};

export type RecommendationAPIResponse = {
  user_id: string;
  recommendations: RecommendationBundle[];
  total: number;
  profile_snapshot: ProfileSnapshot;
};
