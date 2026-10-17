import { Check, LoaderCircle, Minus, X } from 'lucide-react';
import type { AnalysisState } from '../model/types';

export function ProcessingSteps({ state }: { state: AnalysisState }) {
  const current = state.phase === 'uploading' ? 0 : state.phase === 'processing' ? 1 : -1;
  const complete = state.phase === 'done';
  const stopped = state.phase === 'failed' || state.phase === 'cancelled';
  return <ol className="processing-steps" aria-label="Этапы проверки">
    {['Загрузка записи', 'Анализ событий', 'Просмотр результата'].map((label, index) => {
      const done = complete || current > index;
      return <li key={label} className={current === index ? 'current' : done ? 'complete' : ''}>
        <span className="step-icon">{stopped ? (state.phase === 'failed' ? <X size={12} /> : <Minus size={12} />) : done ? <Check size={12} /> : current === index ? <LoaderCircle size={13} className="spin" /> : index + 1}</span>
        <span>{label}</span><small>{stopped ? (state.phase === 'failed' ? 'Прервано' : 'Отменено') : done ? 'Готово' : current === index ? 'В процессе' : 'Ожидание'}</small>
      </li>;
    })}
  </ol>;
}
