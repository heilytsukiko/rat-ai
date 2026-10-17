import { useId, useRef, useState } from 'react';
import { Upload } from 'lucide-react';

interface FileDropzoneProps {
  onSelect: (files: File[]) => void;
  disabled: boolean;
  invalid: boolean;
}

export function FileDropzone({ onSelect, disabled, invalid }: FileDropzoneProps) {
  const id = useId();
  const depth = useRef(0);
  const [dragging, setDragging] = useState(false);
  return <div className={`dropzone ${dragging ? 'is-dragging' : ''} ${invalid ? 'is-invalid' : ''} ${disabled ? 'is-disabled' : ''}`}
    onDragEnter={event => { event.preventDefault(); if (!disabled) { depth.current++; setDragging(true); } }}
    onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = disabled ? 'none' : 'copy'; }}
    onDragLeave={event => { event.preventDefault(); depth.current = Math.max(0, depth.current - 1); if (!depth.current) setDragging(false); }}
    onDrop={event => {
      event.preventDefault(); depth.current = 0; setDragging(false);
      if (!disabled) onSelect(Array.from(event.dataTransfer.files));
    }}>
    <input id={id} className="file-input" type="file" accept=".mov,video/quicktime" disabled={disabled}
      aria-label="Выбрать видео MOV" aria-invalid={invalid} aria-describedby={`file-help${invalid ? ' file-error' : ''}`}
      onChange={event => { const files = Array.from(event.target.files ?? []); if (files.length) onSelect(files); event.target.value = ''; }} />
    <Upload size={29} strokeWidth={1.5} aria-hidden="true" />
    <strong>{dragging ? 'Отпустите файл для выбора' : 'Перетащите видео сюда'}</strong>
    <span>или <label className="file-select-label" htmlFor={id}>выберите файл</label></span>
    <small id="file-help">.mov · до 100 МБ · один файл</small>
  </div>;
}
