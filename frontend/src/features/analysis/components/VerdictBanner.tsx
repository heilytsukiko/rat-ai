import { CircleCheck, TriangleAlert } from 'lucide-react';
import type { AnalysisResult } from '../model/types';

export function VerdictBanner({ result }: { result: AnalysisResult }) {
  const clean = result.verdict === 'no_violation';
  return <div className={`verdict-banner ${clean ? 'is-clean' : ''}`} role="status">
    {clean ? <CircleCheck size={23} /> : <TriangleAlert size={23} />}
    <div><h3>{clean ? 'Нарушение не обнаружено' : 'Обнаружены признаки возможных нарушений'}</h3>
      <p>{result.isDemo ? 'Демонстрационный результат' : 'Анализ завершён'} · {clean ? 'События отсутствуют' : `Событий для проверки: ${result.violations.length}`}</p>
    </div>
  </div>;
}
