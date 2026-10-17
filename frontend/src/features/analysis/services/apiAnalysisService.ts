import type { AnalysisResult, AnalysisService } from '../model/types';

function errorMessage(status: number): string {
  return ({ 401: 'Нет доступа к API. Проверьте ключ на сервере.', 413: 'Размер видео превышает лимит сервера.', 415: 'Сервер не поддерживает формат видео.', 422: 'Сервер отклонил видео. Проверьте файл.', 429: 'Очередь заполнена. Повторите попытку позже.' } as Record<number, string>)[status]
    ?? `Сервис недоступен (HTTP ${status}). Повторите попытку.`;
}

function upload(file: File, signal: AbortSignal, progress: (value: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => { xhr.abort(); reject(new DOMException('Операция отменена', 'AbortError')); };
    if (signal.aborted) { abort(); return; }
    signal.addEventListener('abort', abort, { once: true });
    xhr.open('POST', '/api/v1/analyze');
    xhr.responseType = 'json';
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) progress(Math.round(event.loaded / event.total * 100));
    };
    xhr.onloadend = () => signal.removeEventListener('abort', abort);
    xhr.onerror = () => reject(new Error('Не удалось связаться с сервером. Проверьте подключение.'));
    xhr.onabort = () => reject(new DOMException('Операция отменена', 'AbortError'));
    xhr.onload = () => {
      if (xhr.status !== 202) { reject(new Error(errorMessage(xhr.status))); return; }
      const jobId = xhr.response?.job_id;
      if (typeof jobId !== 'string' || !jobId) { reject(new Error('Сервер вернул некорректный номер задачи.')); return; }
      progress(100);
      resolve(jobId);
    };
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

function delay(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('Операция отменена', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 1500);
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  });
}

type ApiJob = {
  job_id: string;
  status: 'processing' | 'done' | 'failed';
  verdict?: 'violation' | 'no_violation';
  duration_sec?: number;
  violations?: {
    type: string; timecode_sec: number; description: string;
    source?: { title: string; section?: string; url: string };
    evidence_frames: { timestamp_sec: number; image_base64: string }[];
  }[];
};

export function mapResult(job: ApiJob, fileName: string): AnalysisResult {
  if (!['violation', 'no_violation'].includes(job.verdict ?? '') || !Array.isArray(job.violations)
      || (job.verdict === 'violation') !== Boolean(job.violations.length)) {
    throw new Error('Сервер вернул некорректный результат анализа.');
  }
  return {
    jobId: job.job_id, fileName, verdict: job.verdict!, durationSec: job.duration_sec, isDemo: false,
    violations: job.violations.map((event, index) => ({
      id: `${job.job_id}-${index}`, type: event.type, timecodeSec: event.timecode_sec,
      description: event.description, source: event.source,
      evidenceFrames: event.evidence_frames.map((frame, frameIndex) => ({
        id: `${job.job_id}-${index}-${frameIndex}`, timestampSec: frame.timestamp_sec,
        imageUrl: `data:image/jpeg;base64,${frame.image_base64}`, alt: `Кадр: ${event.type}`,
      })),
    })),
  };
}

export const apiAnalysisService: AnalysisService = {
  async analyze({ file }, { signal, onUploadProgress, onProcessing }) {
    if (!file) throw new Error('Выберите видео для проверки.');
    const jobId = await upload(file, signal, onUploadProgress);
    onProcessing();
    for (;;) {
      const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, { signal, cache: 'no-store' });
      if (!response.ok) throw new Error(errorMessage(response.status));
      const job: ApiJob = await response.json();
      if (job.job_id !== jobId) throw new Error('Сервер вернул другую задачу.');
      if (job.status === 'failed') throw new Error('Не удалось обработать видео. Проверьте настройки локальной модели и повторите попытку.');
      if (job.status === 'done') return mapResult(job, file.name);
      if (job.status !== 'processing') throw new Error('Сервер вернул неизвестный статус задачи.');
      await delay(signal);
    }
  },
};
