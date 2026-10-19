export type DemoScenario = 'violation' | 'no_violation' | 'failed';
export type Verdict = 'violation' | 'no_violation' | 'review_required';

export interface RegulatorySource {
  title: string;
  section?: string;
  url: string;
}

export interface EvidenceFrame {
  id: string;
  timestampSec: number;
  imageUrl: string;
  alt: string;
}

export interface ObservationReport {
  summary: string;
  actions: { timestamp_sec: number; description: string }[];
  object_description: string;
  assessment: 'possible_improper_transfer' | 'ordinary_exchange' | 'no_transfer_visible' | 'unclear';
  reason: string;
  limitations: string[];
  sampled_timestamps_sec: number[];
}

export interface AnalysisEvent {
  observationReport?: ObservationReport;
  id: string;
  type: string;
  timecodeSec: number;
  description: string;
  evidenceFrames: EvidenceFrame[];
  source?: RegulatorySource;
  detector?: { score: number; threshold: number; hasObservation: boolean };
}

export interface AnalysisResult {
  jobId: string;
  verdict: Verdict;
  violations: AnalysisEvent[];
  fileName: string;
  durationSec?: number;
  isDemo: boolean;
  experimental?: boolean;
}

export interface AnalysisInput {
  file?: File;
  scenario?: DemoScenario;
}

export interface AnalysisService {
  analyze(
    input: AnalysisInput,
    options: {
      signal: AbortSignal;
      onUploadProgress: (percentage: number) => void;
      onProcessing: () => void;
    },
  ): Promise<AnalysisResult>;
}

export type AnalysisState =
  | { phase: 'idle' }
  | { phase: 'preview'; result: AnalysisResult }
  | { phase: 'cancelled'; stage?: 'uploading' | 'processing' }
  | { phase: 'uploading'; progress: number }
  | { phase: 'processing' }
  | { phase: 'done'; result: AnalysisResult }
  | { phase: 'failed'; message: string; stage?: 'uploading' | 'processing' };
