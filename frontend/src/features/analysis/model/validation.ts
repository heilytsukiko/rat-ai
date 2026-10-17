// A UI safeguard, not a limit specified by the competition API.
export const MAX_FILE_SIZE = 100 * 1024 * 1024;

export function validateVideo(file: File): string | null {
  if (!/\.mov$/i.test(file.name)) return 'Поддерживается только формат .mov. Выберите другое видео.';
  if (file.size === 0) return 'Файл пуст. Выберите другую запись.';
  if (file.size > MAX_FILE_SIZE) return 'Размер файла больше 100 МБ. Выберите запись меньшего размера.';
  return null;
}
