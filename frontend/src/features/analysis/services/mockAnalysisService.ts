import type { AnalysisService } from '../model/types';
import { createMockResult } from '../mocks/results';

function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
      reject(new DOMException('Операция отменена', 'AbortError'));
    };
    const timeout = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, ms);
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  });
}

export const mockAnalysisService: AnalysisService = {
  async analyze(input, { signal, onUploadProgress, onProcessing }) {
    // No file bytes are read or transmitted. All results are predefined.
    for (const percentage of [20, 45, 75, 100]) {
      await delay(250, signal);
      onUploadProgress(percentage);
    }
    onProcessing();
    await delay(1800, signal);
    if (input.scenario === 'failed') {
      throw new Error('Не удалось обработать запись. Это демонстрация ошибки сервиса. Повторите попытку или выберите другой сценарий.');
    }
    const result = createMockResult(input.scenario ?? 'violation', input.file?.name);
    return input.file ? { ...result, durationSec: undefined } : result;
  },
};
