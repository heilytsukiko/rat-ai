import { ObservationDetails } from './ObservationDetails';
import { ExternalLink, BookOpen } from 'lucide-react';
import { SectionLabel } from '../../../components/ui/SectionLabel';
import type { AnalysisEvent } from '../model/types';

export function EventDetails({ event, index, total, experimental = false }: { event: AnalysisEvent; index: number; total: number; experimental?: boolean }) {
  const source = event.source;
  // API-provided links must never use executable protocols.
  const safeUrl = source && /^https?:\/\//i.test(source.url) ? source.url : undefined;
  return <article className="event-details" aria-label="Описание выбранного события">
    <SectionLabel>{experimental ? 'Фрагмент' : 'Событие'} {String(index + 1).padStart(2, '0')} / {String(total).padStart(2, '0')}</SectionLabel>
    <h3>{event.type}</h3><span className="review-tag">{experimental ? (event.observationReport ? 'Предварительный разбор ИИ' : 'Автоматическая отметка') : 'Требует проверки человеком'}</span><p className="event-description">{event.description}</p>
    {event.observationReport && <ObservationDetails report={event.observationReport} />}
    {experimental ? <section className="regulatory-source">
      {!event.observationReport && <><h4>{event.detector?.hasObservation ? 'Сверьте описание с записью' : 'Почему нет описания действия?'}</h4>
      <p>{event.detector?.hasObservation
        ? 'Это предварительное описание модели. Проверьте его по видео; применимое правило и нормативный источник пока не определены.'
        : 'В этом запуске подключён только поиск подозрительных моментов. Для описания действий нужен дополнительный модуль анализа видео; сейчас он не подключён. Повторная загрузка того же файла это не исправит.'}</p></>}
      {event.detector && <details><summary>Технические данные детектора</summary><p>Балл: {event.detector.score.toFixed(3)} · порог отбора: {event.detector.threshold.toFixed(3)}. Это внутренний балл модели, а не вероятность нарушения. Превышение порога лишь добавляет фрагмент в список просмотра.</p></details>}
    </section> : <section className="regulatory-source"><div className="source-label"><BookOpen size={14} /><SectionLabel>Нормативное основание</SectionLabel></div>
      {source ? <><h4>{source.title}</h4>{source.section && <p>{source.section}</p>}{safeUrl && <a href={safeUrl} target="_blank" rel="noopener noreferrer">Открыть источник<ExternalLink size={13} /></a>}</>
        : <><h4>Источник пока не привязан</h4><p>Здесь появятся название документа РК, пункт и ссылка на источник, подтверждающий правило.</p></>}
      <details><summary>Как обоснован вывод</summary><p>Наблюдаемое действие → применимое правило → документ и пункт. Основание предоставлено моделью и требует проверки человеком.</p></details>
    </section>}
  </article>;
}
