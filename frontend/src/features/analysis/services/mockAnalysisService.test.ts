import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockAnalysisService } from './mockAnalysisService';

describe('mockAnalysisService', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  function options() {
    return { signal: new AbortController().signal, onUploadProgress: vi.fn(), onProcessing: vi.fn() };
  }

  it('reports upload and processing before returning clearly marked demo events', async () => {
    const callbacks = options();
    const task = mockAnalysisService.analyze({ scenario: 'violation', file: new File(['video'], 'own.mov') }, callbacks);
    await vi.advanceTimersByTimeAsync(1000);
    expect(callbacks.onUploadProgress).toHaveBeenLastCalledWith(100);
    expect(callbacks.onProcessing).toHaveBeenCalledOnce();
    await vi.runAllTimersAsync();
    const result = await task;
    expect(result.isDemo).toBe(true);
    expect(result.fileName).toBe('own.mov');
    expect(result.durationSec).toBeUndefined();
    expect(result.violations).toHaveLength(2);
    expect(result.violations.every(event => event.source === undefined)).toBe(true);
  });

  it('returns no events for the clean verdict', async () => {
    const task = mockAnalysisService.analyze({ scenario: 'no_violation' }, options());
    await vi.runAllTimersAsync();
    expect(await task).toMatchObject({ verdict: 'no_violation', violations: [] });
  });

  it('surfaces service errors', async () => {
    const task = mockAnalysisService.analyze({ scenario: 'failed' }, options());
    const assertion = expect(task).rejects.toThrow('Не удалось обработать');
    await vi.runAllTimersAsync();
    await assertion;
  });

  it('cancels without reporting further progress', async () => {
    const controller = new AbortController();
    const callbacks = { ...options(), signal: controller.signal };
    const task = mockAnalysisService.analyze({ scenario: 'violation' }, callbacks);
    const assertion = expect(task).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await vi.runAllTimersAsync();
    await assertion;
    expect(callbacks.onUploadProgress).not.toHaveBeenCalled();
    expect(callbacks.onProcessing).not.toHaveBeenCalled();
  });
});
