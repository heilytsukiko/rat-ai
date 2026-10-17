import { describe, expect, it } from 'vitest';
import { formatFileSize, formatTime } from './format';

describe('timecode formatting', () => {
  it('carries rounding across a minute boundary', () => {
    expect(formatTime(59.96, true)).toBe('01:00.0');
    expect(formatTime(119.96, true)).toBe('02:00.0');
    expect(formatTime(27.4, true)).toBe('00:27.4');
    expect(formatTime(59.99)).toBe('00:59');
  });
  it('does not render invalid timestamps', () => {
    for (const time of [-10, NaN, Infinity]) expect(formatTime(time)).toBe('00:00');
  });
});

it('preserves the size of small nonempty files', () => {
  expect(formatFileSize(87)).toBe('87 Б');
  expect(formatFileSize(2048)).toBe('2 КБ');
  expect(formatFileSize(1048576)).toBe('1 МБ');
});
