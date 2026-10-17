import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useAnalysis } from './useAnalysis';
import type { AnalysisService } from '../model/types';
import { createMockResult } from '../mocks/results';

afterEach(() => vi.useRealTimers());

describe('analysis lifecycle regressions', () => {
  it('starts idle without a fabricated analysis', () => {
    const { result } = renderHook(() => useAnalysis());
    expect(result.current.state.phase).toBe('idle');
    expect(result.current.file).toBeNull();
  });

  it('rejects a batch without silently choosing its first file', () => {
    const { result } = renderHook(() => useAnalysis());
    act(() => result.current.selectFiles([new File(['a'], 'a.mov'), new File(['b'], 'b.mov')]));
    expect(result.current.file).toBeNull();
    expect(result.current.fileError).toContain('одну запись');
    expect(result.current.state.phase).toBe('idle');
  });

  it('limits a hung service to five minutes and ignores its later progress', async () => {
    vi.useFakeTimers();
    let options!: Parameters<AnalysisService['analyze']>[1];
    const service: AnalysisService = { analyze: (_input, callbacks) => { options = callbacks; return new Promise(() => {}); } };
    const { result } = renderHook(() => useAnalysis(service));
    act(() => result.current.selectFile(new File(['a'], 'own.mov')));
    act(() => { void result.current.start(); });
    await act(() => vi.advanceTimersByTimeAsync(300_000));
    expect(options.signal.aborted).toBe(true);
    expect(result.current.state).toMatchObject({ phase: 'failed', message: expect.stringContaining('5 минут') });
    act(() => options.onProcessing());
    expect(result.current.state.phase).toBe('failed');
  });

  it('cleans up the request and timeout when the page unmounts', () => {
    vi.useFakeTimers();
    let signal!: AbortSignal;
    const service: AnalysisService = { analyze: (_input, options) => { signal = options.signal; return new Promise(() => {}); } };
    const { result, unmount } = renderHook(() => useAnalysis(service));
    act(() => result.current.selectFile(new File(['a'], 'own.mov')));
    act(() => { void result.current.start(); });
    unmount();
    expect(signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('retries the selected video', async () => {
    const analyze = vi.fn<AnalysisService['analyze']>().mockRejectedValueOnce(new Error('Ошибка')).mockResolvedValueOnce(createMockResult('no_violation'));
    const { result } = renderHook(() => useAnalysis({ analyze }));
    act(() => result.current.selectFile(new File(['a'], 'own.mov')));
    await act(() => result.current.start());
    await act(() => result.current.retry());
    expect(analyze.mock.calls[1][0]).toEqual({ file: expect.objectContaining({ name: 'own.mov' }) });
    expect(result.current.state.phase).toBe('done');
  });
});
