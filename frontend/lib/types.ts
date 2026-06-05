export type UserRole = "admin" | "analyst" | "viewer";

export interface User {
  id: string;
  email: string;
  role: UserRole;
}

export interface TopFeature {
  name: string;
  index?: number;
  contribution?: number;
  shap_value?: number;
  feature_value?: number;
}

export interface PredictResult {
  tx_id: string;
  model: string;
  risk_score: number;
  prediction: string;
  confidence: number;
  prob_licit: number;
  prob_illicit: number;
  top_features: TopFeature[];
  prediction_id?: string;
}

export interface UploadResult {
  job_id: string;
  count: number;
  filename: string;
  results: PredictResult[];
  summary: { high_risk: number; mean_risk_score: number };
}

export interface HistoryItem {
  prediction_id: string;
  tx_id: string;
  time_step?: number;
  model: string;
  risk_score: number;
  prediction: string;
  confidence: number;
  prob_licit?: number;
  prob_illicit?: number;
  batch_id?: string;
  case_id?: string;
  case_status?: string;
  created_at: string;
}

export interface HistoryDetail extends HistoryItem {
  top_features: TopFeature[];
  features: number[];
  shap?: {
    id: string;
    method: string;
    top_features: TopFeature[];
    nsamples?: number;
    created_at: string;
  };
}

export interface CaseItem {
  id: string;
  prediction_id: string;
  title: string;
  status: string;
  priority: string;
  assignee_email?: string;
  tx_id?: string;
  risk_score?: number;
  prediction?: string;
  model?: string;
  notes_count: number;
  created_at: string;
  updated_at: string;
}

export interface GraphNode {
  id: string;
  risk_score?: number;
  prediction?: string;
  is_seed?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface DriftPayload {
  kendall_tau?: { static?: Array<{ comparison: string; tau: number; drift: boolean }>; evolve?: Array<{ comparison: string; tau: number; drift: boolean }> };
  alerts?: Array<{ model: string; window: string; tau: number; severity: string }>;
  t43_drift?: {
    static?: {
      f1_drop?: number;
      pre_t43_mean_f1?: number;
      post_t43_mean_f1?: number;
      shutdown_step?: number;
      pre_snapshots?: number[];
      post_snapshots?: number[];
    };
  };
}
