import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useAnalysis } from './useAnalysis';
import type { AnalysisResult, AnalysisService } from '../model/types';
import { createMockResult } from '../mocks/results';

describe('useAnalysis', () => {
  it('does not start a user-file check without a valid video', async () => {
    const service = { analyze: vi.fn() } satisfies AnalysisService;
    const { result } = renderHook(() => useAnalysis(service));
    await act(() => result.current.start());
    expect(service.analyze).not.toHaveBeenCalled();
    expect(result.current.fileError).toContain('Сначала выберите');
  });

  it('preserves the valid file when a replacement is invalid', () => {
    const { result } = renderHook(() => useAnalysis());
    act(() => result.current.selectFile(new File(['video'], 'valid.mov')));
    expect(result.current.state.phase).toBe('idle');
    expect(result.current.file?.name).toBe('valid.mov');
    act(() => result.current.selectFile(new File(['video'], 'wrong.mp4')));
    expect(result.current.file?.name).toBe('valid.mov');
    expect(result.current.fileError).toContain('.mov');
  });

  it('ignores a late result from a cancelled request even if the service ignores abort', async () => {
    let finish!: (value: AnalysisResult) => void;
    let signal: AbortSignal | undefined;
    const service: AnalysisService = { analyze: (_input, options) => {
      signal = options.signal;
      return new Promise(resolve => { finish = resolve; });
    } };
    const { result } = renderHook(() => useAnalysis(service));
    let pending!: Promise<void>;
    act(() => { pending = result.current.start(true); });
    expect(result.current.state.phase).toBe('uploading');
    act(() => result.current.cancel());
    expect(signal?.aborted).toBe(true);
    await act(async () => { finish(createMockResult('violation')); await pending; });
    expect(result.current.state.phase).toBe('cancelled');
  });
});
