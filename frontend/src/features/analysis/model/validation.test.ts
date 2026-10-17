import { describe, expect, it } from 'vitest';
import { MAX_FILE_SIZE, validateVideo } from './validation';

describe('validateVideo', () => {
  it('accepts MOV case-insensitively even when the OS omits its MIME type', () => {
    expect(validateVideo(new File(['video'], 'касса.MOV'))).toBeNull();
  });
  it('rejects an unsupported format and an empty video', () => {
    expect(validateVideo(new File(['video'], 'recording.mp4'))).toContain('.mov');
    expect(validateVideo(new File([], 'empty.mov'))).toContain('пуст');
  });
  it('enforces the configured size boundary', () => {
    const file = new File(['data'], 'large.mov');
    Object.defineProperty(file, 'size', { value: MAX_FILE_SIZE + 1 });
    expect(validateVideo(file)).toContain('100 МБ');
  });
});
