import apiClient from "./api";

export type AnalyticsEventType =
  | "profile_completed"
  | "rec_viewed"
  | "rec_clicked"
  | "feedback_submitted";

export interface AnalyticsEventPayload {
  eventType: AnalyticsEventType;
  userId?: string;
  role?: string;
  score?: number;
  rating?: number;
  relevant?: boolean;
  context?: Record<string, unknown>;
}

export const logAnalyticsEvent = async (payload: AnalyticsEventPayload) => {
  try {
    await apiClient.post("/analytics/events", payload);
  } catch (error) {
    console.warn("Unable to record analytics event", payload, error);
  }
};
