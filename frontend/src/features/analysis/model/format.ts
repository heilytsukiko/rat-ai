export function formatTime(seconds: number, precise = false): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const ticks = Math.floor(safeSeconds * (precise ? 10 : 1) + (precise ? 0.5 : 0));
  const perMinute = precise ? 600 : 60;
  const minutes = Math.floor(ticks / perMinute).toString().padStart(2, '0');
  const remainder = ticks % perMinute;
  return `${minutes}:${precise ? (remainder / 10).toFixed(1).padStart(4, '0') : remainder.toString().padStart(2, '0')}`;
}

export function formatFileSize(bytes: number): string {
  const value = Math.max(0, Number.isFinite(bytes) ? bytes : 0);
  const format = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 });
  if (value < 1024) return `${format.format(value)} Б`;
  if (value < 1024 * 1024) return `${format.format(value / 1024)} КБ`;
  return `${format.format(value / 1024 / 1024)} МБ`;
}
