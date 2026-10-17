import { useRef, useState } from 'react';
import { Upload } from 'lucide-react';

interface FileDropzoneProps {
  onSelect: (file: File) => void;
  disabled: boolean;
  invalid: boolean;
}

export function FileDropzone({ onSelect, disabled, invalid }: FileDropzoneProps) {
  const input = useRef<HTMLInputElement>(null);
  const depth = useRef(0);
  const [dragging, setDragging] = useState(false);
  return <div className={`dropzone ${dragging ? 'is-dragging' : ''} ${invalid ? 'is-invalid' : ''} ${disabled ? 'is-disabled' : ''}`}
    onDragEnter={event => { event.preventDefault(); if (!disabled) { depth.current++; setDragging(true); } }}
    onDragOver={event => event.preventDefault()}
    onDragLeave={event => { event.preventDefault(); depth.current = Math.max(0, depth.current - 1); if (!depth.current) setDragging(false); }}
    onDrop={event => {
      event.preventDefault(); depth.current = 0; setDragging(false);
      if (!disabled && event.dataTransfer.files[0]) onSelect(event.dataTransfer.files[0]);
    }}>
    <input ref={input} className="sr-only" type="file" accept=".mov,video/quicktime" disabled={disabled}
      aria-label="Выбрать видео MOV" tabIndex={-1}
      onChange={event => { const next = event.target.files?.[0]; if (next) onSelect(next); event.target.value = ''; }} />
    <Upload size={29} strokeWidth={1.5} aria-hidden="true" />
    <strong>Перетащите видео сюда</strong>
    <span>или <button className="text-button" disabled={disabled} onClick={() => input.current?.click()}>выберите файл</button></span>
    <small>.mov · до 100 МБ · один файл</small>
  </div>;
}
