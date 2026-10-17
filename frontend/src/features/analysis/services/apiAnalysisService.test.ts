import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiAnalysisService, mapResult } from './apiAnalysisService';

class FakeXHR {
  static last: FakeXHR;
  status = 202;
  response = { job_id: 'job-1' };
  responseType = '';
  upload: { onprogress?: (event: { lengthComputable: boolean; loaded: number; total: number }) => void } = {};
  onload?: () => void;
  onloadend?: () => void;
  onerror?: () => void;
  onabort?: () => void;
  open = vi.fn();
  send = vi.fn(() => {
    this.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 2 });
    queueMicrotask(() => { this.onload?.(); this.onloadend?.(); });
  });
  abort = vi.fn(() => { this.onabort?.(); this.onloadend?.(); });
  constructor() { FakeXHR.last = this; }
}

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

function setup() {
  vi.stubGlobal('XMLHttpRequest', FakeXHR);
  const controller = new AbortController();
  const options = { signal: controller.signal, onUploadProgress: vi.fn(), onProcessing: vi.fn() };
  const input = { file: new File(['video'], 'own.mov'), scenario: 'violation' as const };
  return { controller, options, input };
}

describe('API analysis service', () => {
  it('uploads multipart video, polls processing and maps a real result', async () => {
    vi.useFakeTimers();
    const { options, input } = setup();
    const fetcher = vi.fn().mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'job-1', status: 'processing' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'job-1', status: 'done', verdict: 'no_violation', violations: [], duration_sec: 15 }) });
    vi.stubGlobal('fetch', fetcher);
    const pending = apiAnalysisService.analyze(input, options);
    await vi.advanceTimersByTimeAsync(1500);
    const result = await pending;
    expect(FakeXHR.last.open).toHaveBeenCalledWith('POST', '/api/v1/analyze');
    expect(FakeXHR.last.send.mock.calls[0]).toBeDefined();
    expect(options.onUploadProgress).toHaveBeenCalledWith(50);
    expect(options.onProcessing).toHaveBeenCalledOnce();
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(result).toMatchObject({ jobId: 'job-1', fileName: 'own.mov', isDemo: false, durationSec: 15, verdict: 'no_violation' });
  });

  it('maps source, timestamps and JPEG evidence', () => {
    const source = { title: 'Document', section: '1', url: 'https://adilet.zan.kz/test' };
    const result = mapResult({ job_id: 'j', status: 'done', verdict: 'violation', violations: [{ type: 'Event', timecode_sec: 4, description: 'Observed action', source, evidence_frames: [{ timestamp_sec: 4, image_base64: '/9j/' }] }] }, 'a.mov');
    expect(result.violations[0]).toMatchObject({ timecodeSec: 4, source, evidenceFrames: [{ timestampSec: 4, imageUrl: 'data:image/jpeg;base64,/9j/' }] });
  });

  it('shows model failure instead of a successful verdict', async () => {
    const { options, input } = setup();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ job_id: 'job-1', status: 'failed' }) }));
    await expect(apiAnalysisService.analyze(input, options)).rejects.toThrow('локальной модели');
  });

  it.each([401, 413, 429])('handles upload rejection %s', async status => {
    const { options, input } = setup();
    const pending = apiAnalysisService.analyze(input, options);
    FakeXHR.last.status = status;
    await expect(pending).rejects.toThrow();
  });

  it('aborts polling without another request', async () => {
    vi.useFakeTimers();
    const { options, input, controller } = setup();
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ job_id: 'job-1', status: 'processing' }) });
    vi.stubGlobal('fetch', fetcher);
    const pending = apiAnalysisService.analyze(input, options);
    const rejection = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await rejection;
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetcher).toHaveBeenCalledOnce();
  });
});

it.each([
  ['network', new TypeError('Failed to fetch'), 'подключение'],
  ['json', new SyntaxError('Unexpected token <'), 'неверном формате'],
])('turns a polling %s failure into an actionable message', async (kind, error, expected) => {
  const { options, input } = setup();
  const fetcher = kind === 'network' ? vi.fn().mockRejectedValue(error)
    : vi.fn().mockResolvedValue({ ok: true, json: async () => { throw error; } });
  vi.stubGlobal('fetch', fetcher);
  await expect(apiAnalysisService.analyze(input, options)).rejects.toThrow(expected);
});

it.each([null, [], { job_id: 'different', status: 'processing' }])('rejects malformed polling envelopes: %j', value => {
  const { options, input } = setup();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => value }));
  return expect(apiAnalysisService.analyze(input, options)).rejects.toThrow('Сервер вернул');
});
