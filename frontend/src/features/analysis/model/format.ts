export function formatTime(seconds: number, precise = false): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
  const remainder = seconds % 60;
  return `${minutes}:${precise ? remainder.toFixed(1).padStart(4, '0') : Math.floor(remainder).toString().padStart(2, '0')}`;
}

export function formatFileSize(bytes: number): string {
  return `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(bytes / 1024 / 1024)} МБ`;
}
