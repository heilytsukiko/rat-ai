import { useState } from 'react';
import { Check, ChevronLeft, ChevronRight, FileSearch, Info, LoaderCircle, RotateCcw, TriangleAlert } from 'lucide-react';
import { SectionLabel } from '../../../components/ui/SectionLabel';
import type { AnalysisResult, AnalysisState } from '../model/types';
import { formatTime } from '../model/format';
import { EventDetails } from './EventDetails';
import { EventList } from './EventList';
import { EvidenceViewer } from './EvidenceViewer';
import { VerdictBanner } from './VerdictBanner';

function CompletedAnalysis({ result, preview }: { result: AnalysisResult; preview: boolean }) {
  const [selectedId, setSelectedId] = useState(result.violations[0]?.id ?? '');
  const selectedEvent = result.violations.find(event => event.id === selectedId) ?? result.violations[0];
  const selectedIndex = selectedEvent ? result.violations.indexOf(selectedEvent) : 0;
  return <>
    <div className="result-heading">
      <div><SectionLabel>02 / {preview ? 'Образец результата' : 'Результат проверки'}</SectionLabel>
        <p className="result-file" title={result.fileName}>{result.fileName}{result.durationSec ? ` · ${formatTime(result.durationSec)}` : ''}</p></div>
      <span className={`completion-label ${preview ? 'is-preview' : ''}`}>{preview ? <Info size={13} /> : <Check size={13} />}{preview ? 'Пример' : 'Готово'}</span>
    </div>
    {result.isDemo && <div className="demo-result-note"><Info size={16} /><p><strong>{preview ? 'Так будет выглядеть результат.' : 'Это результат демонстрации.'}</strong> События и кадры условные. {preview ? 'Запустите пример или выберите видео слева.' : 'Выбранный файл не анализировался.'}</p></div>}
    <VerdictBanner result={result} />
    {result.verdict === 'no_violation' ? <div className="empty-result"><div className="empty-icon success"><Check size={28} /></div><h3>События отсутствуют</h3><p>В записи не обнаружены признаки возможных нарушений. Результат предназначен для проверки человеком.</p></div>
      : selectedEvent ? <>
        <div className="review-toolbar"><h3>Разбор событий <span>{result.violations.length}</span></h3>
          <div className="event-pagination"><span aria-live="polite">{selectedIndex + 1} из {result.violations.length}</span>
            <button className="icon-button" aria-label="Предыдущее событие" disabled={selectedIndex === 0} onClick={() => setSelectedId(result.violations[selectedIndex - 1].id)}><ChevronLeft size={18} /></button>
            <button className="icon-button" aria-label="Следующее событие" disabled={selectedIndex === result.violations.length - 1} onClick={() => setSelectedId(result.violations[selectedIndex + 1].id)}><ChevronRight size={18} /></button>
          </div>
        </div>
        <div className="review-grid"><div className="review-media">
          <EvidenceViewer key={selectedEvent.id} event={selectedEvent} events={result.violations} durationSec={result.durationSec} isDemo={result.isDemo} onSelectEvent={setSelectedId} />
          <EventList events={result.violations} selectedId={selectedEvent.id} onSelect={setSelectedId} />
          <p className="evidence-caption">Нажмите на кадр, чтобы рассмотреть детали.</p>
        </div><EventDetails event={selectedEvent} index={selectedIndex} total={result.violations.length} /></div>
      </> : <div className="missing-evidence" role="status">Сервис не предоставил описания событий. Обоснование результата недоступно.</div>}
  </>;
}

export function AnalysisResults({ state, onRetry }: { state: AnalysisState; onRetry: () => void }) {
  const busy = state.phase === 'uploading' || state.phase === 'processing';
  const cancelled = state.phase === 'cancelled';
  return <section id="analysis-result" tabIndex={-1} className="results-panel panel" aria-label="Результат проверки" aria-busy={busy}>
    {state.phase === 'done' || state.phase === 'preview' ? <CompletedAnalysis result={state.result} preview={state.phase === 'preview'} /> : <>
      <SectionLabel>02 / Результат проверки</SectionLabel>
      <div className="empty-result" role={state.phase === 'failed' ? 'alert' : 'status'}>
        <div className={`empty-icon ${state.phase === 'failed' ? 'error' : ''}`}>{busy ? <LoaderCircle size={29} className="spin" /> : state.phase === 'failed' ? <TriangleAlert size={29} /> : <FileSearch size={29} />}</div>
        <h3>{state.phase === 'uploading' ? 'Подготавливаем запись' : state.phase === 'processing' ? 'Проверяем события' : state.phase === 'failed' ? 'Проверка не завершена' : cancelled ? 'Проверка отменена' : 'Всё готово к проверке'}</h3>
        <p>{state.phase === 'failed' ? state.message : busy ? 'Видео обрабатывается на сервере. Анализ может занять до 5 минут.' : cancelled ? 'Ожидание прекращено. Задача на сервере может продолжаться. Вы можете запустить новую проверку.' : 'Загрузите видео и запустите проверку слева. Здесь появятся вердикт, события и кадры-доказательства.'}</p>
        {state.phase === 'uploading' && <div className="result-progress"><progress max="100" value={state.progress} aria-label="Подготовка записи" /><span>{state.progress}%</span></div>}
        {state.phase === 'processing' && <div className="processing-indicator"><span />Анализируем запись</div>}
        {state.phase === 'failed' && <button className="button button-secondary" onClick={onRetry}><RotateCcw size={15} />Повторить попытку</button>}
      </div>
    </>}
  </section>;
}
