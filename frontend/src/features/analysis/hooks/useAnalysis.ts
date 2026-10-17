import { useEffect, useRef, useState } from 'react';
import type { AnalysisService, AnalysisState, DemoScenario } from '../model/types';
import { validateVideo } from '../model/validation';
import { createMockResult } from '../mocks/results';
import { mockAnalysisService } from '../services/mockAnalysisService';

export function useAnalysis(service: AnalysisService = mockAnalysisService) {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [scenario, setScenario] = useState<DemoScenario>('violation');
  const [state, setState] = useState<AnalysisState>({ phase: 'done', result: createMockResult('violation') });
  const activeRequest = useRef<AbortController | null>(null);
  const lastRunWasExample = useRef(false);
  const busy = state.phase === 'uploading' || state.phase === 'processing';

  useEffect(() => () => activeRequest.current?.abort(), []);

  function cancel() {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setState({ phase: 'idle' });
  }

  function selectFile(next: File) {
    cancel();
    const error = validateVideo(next);
    setFileError(error);
    setFile(error ? null : next);
  }

  function clearFile() {
    cancel();
    setFile(null);
    setFileError(null);
  }

  async function start(useExample = false) {
    if (!useExample && !file) {
      setFileError('Сначала выберите .mov-видео или запустите готовый пример.');
      return;
    }
    lastRunWasExample.current = useExample;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setFileError(null);
    setState({ phase: 'uploading', progress: 0 });
    try {
      const result = await service.analyze({ file: useExample ? undefined : file ?? undefined, scenario }, {
        signal: controller.signal,
        onUploadProgress: progress => {
          if (activeRequest.current === controller) setState({ phase: 'uploading', progress });
        },
        onProcessing: () => {
          if (activeRequest.current === controller) setState({ phase: 'processing' });
        },
      });
      if (activeRequest.current === controller) setState({ phase: 'done', result });
    } catch (error) {
      if (controller.signal.aborted || activeRequest.current !== controller) return;
      setState({ phase: 'failed', message: error instanceof Error ? error.message : 'Не удалось выполнить проверку. Попробуйте ещё раз.' });
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null;
    }
  }

  const retry = () => start(lastRunWasExample.current);

  return { file, fileError, scenario, setScenario, state, busy, selectFile, clearFile, start, retry, cancel };
}
