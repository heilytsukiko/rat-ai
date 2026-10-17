import type { AnalysisService } from '../model/types';

import { isRecord, mapResult } from './apiResult';
export { mapResult } from './apiResult';

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
      if (event.lengthComputable && event.total > 0 && !signal.aborted) progress(Math.min(100, Math.max(0, Math.round(event.loaded / event.total * 100))));
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
    try { xhr.send(form); } catch {
      signal.removeEventListener('abort', abort);
      reject(new Error('Не удалось отправить видео. Проверьте подключение к серверу.'));
    }
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

export const apiAnalysisService: AnalysisService = {
  async analyze({ file }, { signal, onUploadProgress, onProcessing }) {
    if (!file) throw new Error('Выберите видео для проверки.');
    const jobId = await upload(file, signal, onUploadProgress);
    signal.throwIfAborted();
    onProcessing();
    for (;;) {
      let response: Response;
      try {
        response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, { signal, cache: 'no-store' });
      } catch (error) {
        if (signal.aborted) throw error;
        throw new Error('Не удалось получить статус проверки. Проверьте подключение к серверу.');
      }
      if (!response.ok) throw new Error(errorMessage(response.status));
      let job: unknown;
      try { job = await response.json(); } catch {
        throw new Error('Сервер вернул ответ в неверном формате. Проверьте подключение API.');
      }
      if (!isRecord(job)) throw new Error('Сервер вернул некорректный статус задачи.');
      if (job.job_id !== jobId) throw new Error('Сервер вернул другую задачу.');
      if (job.status === 'failed') throw new Error('Не удалось обработать видео. Проверьте настройки локальной модели и повторите попытку.');
      if (job.status === 'done') return mapResult(job, file.name);
      if (job.status !== 'processing') throw new Error('Сервер вернул неизвестный статус задачи.');
      await delay(signal);
    }
  },
};
