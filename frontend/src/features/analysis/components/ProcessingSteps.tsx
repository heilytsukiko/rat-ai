import { Check, LoaderCircle, Minus, X } from 'lucide-react';
import type { AnalysisState } from '../model/types';

export function ProcessingSteps({ state }: { state: AnalysisState }) {
  const stopped = state.phase === 'failed' || state.phase === 'cancelled';
  const phase = stopped ? state.stage : state.phase;
  const current = phase === 'uploading' ? 0 : phase === 'processing' ? 1 : -1;
  return <ol className="processing-steps" aria-label="Этапы проверки">
    {['Загрузка записи', 'Анализ событий', 'Просмотр результата'].map((label, index) => {
      const done = state.phase === 'done' || current > index;
      const interrupted = stopped && current === index;
      const active = !stopped && current === index;
      return <li key={label} className={active ? 'current' : done ? 'complete' : ''}>
        <span className="step-icon">{done ? <Check size={12} /> : interrupted ? (state.phase === 'failed' ? <X size={12} /> : <Minus size={12} />) : active ? <LoaderCircle size={13} className="spin" /> : index + 1}</span>
        <span>{label}</span><small>{done ? 'Готово' : interrupted ? (state.phase === 'failed' ? 'Ошибка' : 'Нет статуса') : active ? 'В процессе' : 'Ожидание'}</small>
      </li>;
    })}
  </ol>;
}
