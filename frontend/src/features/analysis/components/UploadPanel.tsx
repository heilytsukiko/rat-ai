import { ArrowRight, FileVideo, LoaderCircle, LockKeyhole, X } from 'lucide-react';
import { SectionLabel } from '../../../components/ui/SectionLabel';
import type { AnalysisState } from '../model/types';
import { formatFileSize } from '../model/format';
import { FileDropzone } from './FileDropzone';
import { ProcessingSteps } from './ProcessingSteps';

interface UploadPanelProps {
  file: File | null;
  error: string | null;
  state: AnalysisState;
  busy: boolean;
  onSelect: (files: File[]) => void;
  onClear: () => void;
  onStart: () => void;
  onCancel: () => void;
}

export function UploadPanel({ file, error, state, busy, onSelect, onClear, onStart, onCancel }: UploadPanelProps) {
  const uploading = state.phase === 'uploading';
  return <aside className="upload-panel panel" aria-labelledby="upload-title">
    <SectionLabel>01 / Загрузка</SectionLabel>
    <h2 id="upload-title">Новая проверка</h2><p className="panel-subtitle">Выберите запись работы кассы</p>
    <FileDropzone onSelect={onSelect} disabled={busy} invalid={Boolean(error)} />
    {file ? <div className="selected-file"><FileVideo size={20} /><div><strong title={file.name}>{file.name}</strong><small>{formatFileSize(file.size)} · MOV</small></div>
      <button className="icon-button" aria-label="Убрать выбранный файл" onClick={onClear} disabled={busy}><X size={16} /></button></div>
      : <p className="file-placeholder">Файл ещё не выбран</p>}
    {error && <p id="file-error" className="field-error" role="alert">{error}</p>}
    <button className="button button-primary upload-button" onClick={onStart} disabled={busy}>
      {busy ? <><LoaderCircle className="spin" size={16} />{uploading ? 'Загрузка…' : 'Обработка…'}</> : <>Проверить в демо<ArrowRight size={16} /></>}
    </button>
    {busy && <div className="upload-progress">
      {uploading ? <><div className="progress-caption"><span>Демо: загрузка</span><span>{state.progress}%</span></div><progress max="100" value={state.progress} aria-label="Прогресс демонстрационной загрузки" /></> : <p className="muted">Демо: поиск событий в записи…</p>}
      <button className="text-button" onClick={onCancel}>Отменить проверку</button>
    </div>}
    {(state.phase === 'done' || state.phase === 'failed' || state.phase === 'cancelled' || busy) && <ProcessingSteps state={state} />}
    {state.phase === 'done' && <a className="result-shortcut" href="#analysis-result">Посмотреть результат<ArrowRight size={15} /></a>}
    {state.phase === 'cancelled' && <p className="cancel-note" role="status">Проверка отменена. Можно запустить снова.</p>}
    <div className="privacy-note"><LockKeyhole size={14} /><p>В этом прототипе файл остаётся на устройстве. Загрузка и обработка имитируются.</p></div>
  </aside>;
}
