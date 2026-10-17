import { useEffect, useRef, useState } from 'react';
import type { AnalysisService, AnalysisState } from '../model/types';
import { validateVideo } from '../model/validation';
import { apiAnalysisService } from '../services/apiAnalysisService';

export function useAnalysis(service: AnalysisService = apiAnalysisService) {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [state, setState] = useState<AnalysisState>({ phase: 'idle' });
  const activeRequest = useRef<AbortController | null>(null);
  const requestTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const busy = state.phase === 'uploading' || state.phase === 'processing';

  useEffect(() => () => {
    activeRequest.current?.abort();
    clearTimeout(requestTimeout.current);
    activeRequest.current = null;
  }, []);

  function cancel() {
    activeRequest.current?.abort();
    clearTimeout(requestTimeout.current);
    activeRequest.current = null;
    setState(previous => ({ phase: 'cancelled', stage: previous.phase === 'uploading' || previous.phase === 'processing' ? previous.phase : undefined }));
  }

  function selectFiles(files: File[]) {
    if (files.length !== 1) {
      setFileError('Выберите одну запись. Пакетная загрузка пока не поддерживается.');
      return;
    }
    selectFile(files[0]);
  }

  function selectFile(next: File) {
    const error = validateVideo(next);
    setFileError(error);
    if (error) return;
    cancel();
    setState({ phase: 'idle' });
    setFile(next);
  }

  function clearFile() {
    cancel();
    setFile(null);
    setFileError(null);
    setState({ phase: 'idle' });
  }

  async function start() {
    if (!file) {
      setFileError('Сначала выберите .mov-видео.');
      return;
    }
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setFileError(null);
    setState({ phase: 'uploading', progress: 0 });
    clearTimeout(requestTimeout.current);
    const timeout = setTimeout(() => {
      if (activeRequest.current !== controller) return;
      activeRequest.current = null;
      controller.abort();
      setState(previous => ({ phase: 'failed', message: 'Сервис не ответил за 5 минут. Повторите попытку.', stage: previous.phase === 'uploading' || previous.phase === 'processing' ? previous.phase : undefined }));
    }, 300_000);
    requestTimeout.current = timeout;
    try {
      const result = await service.analyze({ file }, {
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
      setState(previous => ({ phase: 'failed', message: error instanceof Error ? error.message : 'Не удалось выполнить проверку. Попробуйте ещё раз.', stage: previous.phase === 'uploading' || previous.phase === 'processing' ? previous.phase : undefined }));
    } finally {
      clearTimeout(timeout);
      if (activeRequest.current === controller) activeRequest.current = null;
    }
  }

  const retry = () => start();

  return { file, fileError, state, busy, selectFile, selectFiles, clearFile, start, retry, cancel };
}
