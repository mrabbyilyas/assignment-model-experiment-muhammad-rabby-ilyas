export type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  correct: number;
  total: number;
  confusion_matrix: number[][];
};
export type Prediction = {
  review_id: number;
  product_name: string;
  review_text: string;
  actual: string;
  classic: string;
  classic_probability: number;
  llm: string | null;
  llm_latency_ms: number | null;
};
export type Experiment = {
  generated_at: string;
  author: string;
  recommendation: string;
  prompt: string;
  dataset: {
    rows: number;
    unique_texts: number;
    repeated_rows: number;
    products: number;
    labels: Record<string, number>;
    sha256: string;
    source: string;
  };
  split: {
    seed: number;
    train_rows: number;
    test_rows: number;
    train_groups: number;
    test_groups: number;
    text_overlap: number;
    test_labels: Record<string, number>;
    train_labels: Record<string, number>;
    train_ids: number[];
    test_ids: number[];
  };
  classic: {
    name: string;
    metrics: Metrics;
    training_ms: number;
    mean_latency_ms: number;
    p95_latency_ms: number;
    vocabulary_size: number;
  };
  llm: {
    name: string;
    status: string;
    completed: number;
    expected: number;
    metrics: Metrics | null;
    mean_latency_ms: number | null;
    p95_latency_ms: number | null;
    cost_usd: number | null;
    input_tokens: number | null;
    output_tokens: number | null;
    config: {
      model: string;
      provider: string;
      temperature: number;
      max_output_tokens: number;
      reasoning_effort?: string | null;
      prompt_sha256: string;
    };
  };
  diagnostic: {
    accuracy: number;
    overlapping_test_rows: number;
    test_rows: number;
  };
  unique_text_sensitivity: {
    groups: number;
    classic: Metrics;
    llm: Metrics | null;
  };
  features: {
    positif: { term: string; weight: number }[];
    negatif: { term: string; weight: number }[];
  };
  predictions: Prediction[];
  environment: {
    python: string;
    pandas: string;
    scikit_learn: string;
    platform: string;
  };
};
