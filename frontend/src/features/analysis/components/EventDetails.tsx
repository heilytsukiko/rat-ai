import { ExternalLink, BookOpen } from 'lucide-react';
import { SectionLabel } from '../../../components/ui/SectionLabel';
import type { AnalysisEvent } from '../model/types';

export function EventDetails({ event, index, total }: { event: AnalysisEvent; index: number; total: number }) {
  const source = event.source;
  // The future API may provide URLs; never render executable protocols.
  const safeUrl = source && /^https?:\/\//i.test(source.url) ? source.url : undefined;
  return <article className="event-details" aria-label="Описание выбранного события">
    <SectionLabel>Событие {String(index + 1).padStart(2, '0')} / {String(total).padStart(2, '0')}</SectionLabel>
    <h3>{event.type}</h3><span className="review-tag">Требует проверки человеком</span><p className="event-description">{event.description}</p>
    <section className="regulatory-source"><div className="source-label"><BookOpen size={14} /><SectionLabel>Нормативное основание</SectionLabel></div>
      {source ? <><h4>{source.title}</h4>{source.section && <p>{source.section}</p>}{safeUrl && <a href={safeUrl} target="_blank" rel="noopener noreferrer">Открыть источник<ExternalLink size={13} /></a>}</>
        : <><h4>Источник пока не привязан</h4><p>Здесь появятся название документа РК, пункт и ссылка на источник, подтверждающий правило.</p></>}
      <details><summary>Как будет обоснован вывод</summary><p>Наблюдаемое действие → применимое правило → документ и пункт. События в демо условные, нормативные ссылки не выдуманы.</p></details>
    </section>
  </article>;
}
