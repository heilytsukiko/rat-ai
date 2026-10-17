import { describe, expect, it } from 'vitest';
import { INVALID_RESULT, mapResult } from './apiResult';

const valid = () => ({ job_id: 'job', status: 'done', verdict: 'violation', duration_sec: 20,
  violations: [{ type: 'Event', description: 'Observed action', timecode_sec: 4,
    evidence_frames: [{ timestamp_sec: 4, image_base64: '/9j/' }] }] });

describe('API result validation', () => {
  it.each([null, [], {}, { job_id: 'j', status: 'done', verdict: 'no_violation', violations: [null] }])('rejects malformed result %j', value => {
    expect(() => mapResult(value, 'a.mov')).toThrow(INVALID_RESULT);
  });
  it.each([null, 'text', { ...valid().violations[0], type: {} }, { ...valid().violations[0], description: null },
    { ...valid().violations[0], timecode_sec: NaN }, { ...valid().violations[0], timecode_sec: -1 },
    { ...valid().violations[0], timecode_sec: 20 }, { ...valid().violations[0], evidence_frames: null },
    { ...valid().violations[0], evidence_frames: [] }, { ...valid().violations[0], evidence_frames: [null] },
    { ...valid().violations[0], evidence_frames: [{ timestamp_sec: 20, image_base64: '/9j/' }] },
    { ...valid().violations[0], evidence_frames: [{ timestamp_sec: 4, image_base64: 'not jpeg' }] },
    { ...valid().violations[0], source: { title: {}, url: 'https://example.com/' } },
    { ...valid().violations[0], source: { title: 'Source', url: 'javascript:alert(1)' } },
  ])('rejects invalid event data without letting it crash React: %j', event => {
    expect(() => mapResult({ ...valid(), violations: [event] }, 'a.mov')).toThrow(INVALID_RESULT);
  });
  it.each([0, -1, Infinity, '20'])('rejects invalid duration %j', duration => {
    expect(() => mapResult({ ...valid(), duration_sec: duration }, 'a.mov')).toThrow(INVALID_RESULT);
  });
  it('accepts contract results with optional metadata omitted', () => {
    const result = mapResult({ ...valid(), duration_sec: undefined }, 'a.mov');
    expect(result.violations[0].evidenceFrames[0].imageUrl).toBe('data:image/jpeg;base64,/9j/');
    expect(result.violations[0].source).toBeUndefined();
  });
});
