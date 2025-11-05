export type TrajectoryNeighbor = {
  role: string;
  direction: "inbound" | "outbound";
  success_rate: number;
  avg_time_months: number;
  observed_transitions: number;
  common_skills_added: string[];
};

export type TrajectoryResponse = {
  role: string;
  centrality: number | null;
  neighbors: TrajectoryNeighbor[];
  metadata: Record<string, unknown>;
};
