import { formatTime } from '../model/format';
import type { AnalysisEvent, AnalysisResult, RegulatorySource } from '../model/types';

export const INVALID_RESULT = 'Сервер вернул некорректный результат анализа. Повторите проверку.';
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const timestamp = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
function invalid(): never { throw new Error(INVALID_RESULT); }

function sourceFrom(value: unknown): RegulatorySource | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value) || !text(value.title) || !text(value.url) || (value.section !== undefined && !text(value.section))) return invalid();
  try {
    const url = new URL(value.url);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return invalid();
  } catch { return invalid(); }
  return { title: value.title, url: value.url, section: value.section as string | undefined };
}

function jpeg(value: unknown): string {
  if (!text(value) || !/^[A-Za-z0-9+/]+={0,2}$/.test(value) || value.length % 4 !== 0) return invalid();
  try {
    const bytes = atob(value);
    if (bytes.charCodeAt(0) !== 255 || bytes.charCodeAt(1) !== 216) return invalid();
  } catch { return invalid(); }
  return value;
}

// JSON crosses a trust boundary. A TypeScript assertion cannot validate a server response.
export function mapResult(value: unknown, fileName: string): AnalysisResult {
  if (isRecord(value) && value.analysis_mode === 'experimental') return experimentalResult(value, fileName);
  if (!isRecord(value) || !text(value.job_id) || value.status !== 'done'
    || !['violation', 'no_violation'].includes(String(value.verdict)) || !Array.isArray(value.violations)
    || value.violations.length > 100 || (value.verdict === 'violation') !== Boolean(value.violations.length)) return invalid();
  const duration = value.duration_sec;
  if (duration !== undefined && (!timestamp(duration) || duration === 0)) return invalid();
  const inVideo = (time: unknown): time is number => timestamp(time) && (duration === undefined || time < (duration as number));
  const violations: AnalysisEvent[] = value.violations.map((event: unknown, index: number) => {
    if (!isRecord(event) || !text(event.type) || !text(event.description) || !inVideo(event.timecode_sec)
      || !Array.isArray(event.evidence_frames) || event.evidence_frames.length < 1 || event.evidence_frames.length > 3) return invalid();
    return {
      id: `${value.job_id}-${index}`, type: event.type, timecodeSec: event.timecode_sec,
      description: event.description, source: sourceFrom(event.source),
      evidenceFrames: event.evidence_frames.map((frame: unknown, frameIndex: number) => {
        if (!isRecord(frame) || !inVideo(frame.timestamp_sec)) return invalid();
        return { id: `${value.job_id}-${index}-${frameIndex}`, timestampSec: frame.timestamp_sec,
          imageUrl: `data:image/jpeg;base64,${jpeg(frame.image_base64)}`, alt: `Кадр: ${event.type}` };
      }),
    };
  });
  return { jobId: value.job_id, fileName, verdict: value.verdict as AnalysisResult['verdict'],
    durationSec: duration as number | undefined, isDemo: false, violations };
}

function experimentalResult(value: Record<string, unknown>, fileName: string): AnalysisResult {
  const duration = value.duration_sec;
  const threshold = value.threshold;
  if (!text(value.job_id) || value.status !== 'done' || value.experimental !== true
    || value.verdict !== undefined || value.violations !== undefined || !text(value.model)
    || !timestamp(duration) || duration <= 0 || duration > 180
    || !timestamp(threshold) || threshold > 1
    || !Array.isArray(value.candidates) || value.candidates.length > 100) return invalid();
  const violations: AnalysisEvent[] = value.candidates.map((event: unknown, index: number) => {
    if (!isRecord(event) || !timestamp(event.timecode_sec) || event.timecode_sec >= duration
      || !timestamp(event.score) || event.score < threshold || event.score > 1
      || !Array.isArray(event.evidence_frames) || event.evidence_frames.length < 1 || event.evidence_frames.length > 3) return invalid();
    const observation = event.observation_draft;
    if (observation !== undefined && (!text(observation) || observation.length > 1600
      || value.observation_model !== 'Qwen/Qwen2.5-VL-7B-Instruct')) return invalid();
    return { id: `${value.job_id}-${index}`, type: `Момент ${formatTime(event.timecode_sec, true)}`, timecodeSec: event.timecode_sec,
      description: observation
        ? `Предварительное описание ИИ: ${observation}`
        : `Откройте кадр на ${formatTime(event.timecode_sec, true)} и проверьте этот момент в исходном видео. Детектор выделил его для просмотра, но не умеет объяснять действия людей. Это срабатывание само по себе не подтверждает нарушение.`,
      detector: { score: event.score, threshold, hasObservation: Boolean(observation) },
      evidenceFrames: event.evidence_frames.map((frame: unknown, frameIndex: number) => {
        if (!isRecord(frame) || !timestamp(frame.timestamp_sec) || frame.timestamp_sec >= duration
          || Math.abs(frame.timestamp_sec - (event.timecode_sec as number)) > 5) return invalid();
        return { id: `${value.job_id}-${index}-${frameIndex}`, timestampSec: frame.timestamp_sec,
          imageUrl: `data:image/jpeg;base64,${jpeg(frame.image_base64)}`, alt: 'Кадр фрагмента для проверки' };
      }) };
  });
  return { jobId: value.job_id, fileName, verdict: 'review_required', durationSec: duration,
    isDemo: false, experimental: true, violations };
}
