import { useState } from 'react';
import { Check, FileSearch, LoaderCircle, RotateCcw, TriangleAlert } from 'lucide-react';
import { SectionLabel } from '../../../components/ui/SectionLabel';
import type { AnalysisResult, AnalysisState } from '../model/types';
import { formatTime } from '../model/format';
import { EventDetails } from './EventDetails';
import { EventList } from './EventList';
import { EvidenceViewer } from './EvidenceViewer';
import { VerdictBanner } from './VerdictBanner';

function CompletedAnalysis({ result }: { result: AnalysisResult }) {
  const [selectedId, setSelectedId] = useState(result.violations[0]?.id ?? '');
  const selectedEvent = result.violations.find(event => event.id === selectedId) ?? result.violations[0];
  return <>
    <div className="result-heading"><div><SectionLabel>02 / Результат проверки</SectionLabel><p className="result-file" title={result.fileName}>{result.fileName}{result.durationSec ? ` · ${formatTime(result.durationSec)}` : ''}</p></div><span className="completion-label"><Check size={13} />Готово</span></div>
    <VerdictBanner result={result} />
    {result.verdict === 'no_violation' ? <div className="empty-result"><div className="empty-icon success"><Check size={28} /></div><h3>События отсутствуют</h3><p>В этом демонстрационном сценарии признаки нарушений не обнаружены. Можно выбрать другой сценарий или новую запись.</p></div>
      : selectedEvent && <div className="review-grid"><div className="review-media">
        <EvidenceViewer key={selectedEvent.id} event={selectedEvent} events={result.violations} durationSec={result.durationSec} isDemo={result.isDemo} onSelectEvent={setSelectedId} />
        <EventList events={result.violations} selectedId={selectedEvent.id} onSelect={setSelectedId} />
        <p className="evidence-caption">{result.isDemo ? 'Условные события и схемы, не результаты анализа вашего видео.' : 'Нажмите на кадр, чтобы рассмотреть детали.'}</p>
      </div><EventDetails event={selectedEvent} index={result.violations.indexOf(selectedEvent)} total={result.violations.length} /></div>}
  </>;
}

export function AnalysisResults({ state, onRetry }: { state: AnalysisState; onRetry: () => void }) {
  const busy = state.phase === 'uploading' || state.phase === 'processing';
  return <section className="results-panel panel" aria-label="Результат проверки" aria-busy={busy}>
    {state.phase === 'done' ? <CompletedAnalysis result={state.result} /> : <>
      <SectionLabel>02 / Результат проверки</SectionLabel>
      <div className="empty-result" role={state.phase === 'failed' ? 'alert' : 'status'}>
        <div className={`empty-icon ${state.phase === 'failed' ? 'error' : ''}`}>{busy ? <LoaderCircle size={29} className="spin" /> : state.phase === 'failed' ? <TriangleAlert size={29} /> : <FileSearch size={29} />}</div>
        <h3>{state.phase === 'uploading' ? 'Подготавливаем запись' : state.phase === 'processing' ? 'Проверяем события' : state.phase === 'failed' ? 'Проверка не завершена' : 'Всё готово к проверке'}</h3>
        <p>{state.phase === 'failed' ? state.message : busy ? 'Имитируем работу сервиса. Через несколько секунд здесь появится демонстрационный результат.' : 'Выберите видео слева или запустите готовый пример. Здесь появятся вердикт, события и кадры-доказательства.'}</p>
        {state.phase === 'failed' && <button className="button button-secondary" onClick={onRetry}><RotateCcw size={15} />Повторить попытку</button>}
      </div>
    </>}
  </section>;
}
