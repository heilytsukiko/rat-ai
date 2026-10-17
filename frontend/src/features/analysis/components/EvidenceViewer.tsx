import { useState } from 'react';
import { Expand } from 'lucide-react';
import { EvidenceImage } from './EvidenceImage';
import { Modal } from '../../../components/ui/Modal';
import type { AnalysisEvent, EvidenceFrame } from '../model/types';
import { formatTime } from '../model/format';

interface EvidenceViewerProps {
  event: AnalysisEvent;
  events: AnalysisEvent[];
  durationSec?: number;
  isDemo: boolean;
  onSelectEvent: (id: string) => void;
}

export function EvidenceViewer({ event, events, durationSec, isDemo, onSelectEvent }: EvidenceViewerProps) {
  const [expanded, setExpanded] = useState<EvidenceFrame | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const frame = event.evidenceFrames[frameIndex];
  return <div className="evidence-viewer">
    {frame ? <button className="evidence-frame" onClick={() => setExpanded(frame)} aria-label={`Увеличить кадр ${formatTime(frame.timestampSec, true)}`}>
      <EvidenceImage key={frame.imageUrl} src={frame.imageUrl} alt={frame.alt} />
      <span className="camera-label">{isDemo ? 'СХЕМА КАССЫ / ДЕМО' : 'КАДР ИЗ ЗАПИСИ'}</span>
      <span className="expand-label"><Expand size={15} /></span><time className="frame-timestamp">{formatTime(frame.timestampSec, true)}</time>
    </button> : <div className="missing-evidence">Кадры для этого события не предоставлены.</div>}
    {event.evidenceFrames.length > 1 && <div className="frame-picker" aria-label="Кадры события">{event.evidenceFrames.map((item, index) =>
      <button className="button button-secondary" key={item.id} aria-pressed={frameIndex === index} onClick={() => setFrameIndex(index)}>Кадр {index + 1} · {formatTime(item.timestampSec, true)}</button>)}</div>}
    {durationSec !== undefined && durationSec > 0 && Number.isFinite(durationSec) && <><div className="event-timeline" aria-label="Таймкоды событий">{events.map(item => <button key={item.id}
      className={`timeline-marker ${event.id === item.id ? 'is-active' : ''}`} style={{ left: `${Math.max(2, Math.min(98, item.timecodeSec / durationSec * 100))}%` }}
      aria-label={`Перейти к событию ${formatTime(item.timecodeSec, true)}`} title={`${formatTime(item.timecodeSec, true)} — ${item.type}`} onClick={() => onSelectEvent(item.id)} />)}</div>
      <div className="timeline-labels"><span>00:00</span><span>{formatTime(durationSec)}</span></div></>}
    {expanded && <Modal title={`Кадр · ${formatTime(expanded.timestampSec, true)}`} onClose={() => setExpanded(null)}>
      <EvidenceImage key={expanded.imageUrl} className="expanded-frame" src={expanded.imageUrl} alt={expanded.alt} /><p className="muted">{isDemo ? 'Схема для демонстрации интерфейса. Это не кадр из загруженного видео.' : event.description}</p>
    </Modal>}
  </div>;
}
