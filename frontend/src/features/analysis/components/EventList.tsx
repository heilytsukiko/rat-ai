import { ChevronRight } from 'lucide-react';
import type { AnalysisEvent } from '../model/types';
import { formatTime } from '../model/format';

interface EventListProps {
  events: AnalysisEvent[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export function EventList({ events, selectedId, onSelect }: EventListProps) {
  return <div className="event-list" aria-label="События в записи">
    {events.map((event, index) => <button className={`event-item ${event.id === selectedId ? 'is-active' : ''}`} key={event.id}
      aria-pressed={event.id === selectedId} onClick={() => onSelect(event.id)}>
      <time>{formatTime(event.timecodeSec, true)}</time><span><strong>{event.type}</strong><small>Событие {String(index + 1).padStart(2, '0')} · просмотр кадра</small></span><ChevronRight size={15} />
    </button>)}
  </div>;
}
