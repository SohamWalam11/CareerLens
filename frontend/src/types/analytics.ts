export type RoleMetric = {
  role: string;
  views: number;
  clicks: number;
};

export type ScoreMetric = {
  eventType: string;
  averageScore: number;
};

export type FeedbackHeatmapCell = {
  rating: number;
  relevant: boolean;
  count: number;
};

export type AnalyticsTotals = {
  totalEvents: number;
  byType: Record<string, number>;
};

export type AnalyticsSummary = {
  topRoles: RoleMetric[];
  averageScores: ScoreMetric[];
  feedbackHeatmap: FeedbackHeatmapCell[];
  totals: AnalyticsTotals;
};
