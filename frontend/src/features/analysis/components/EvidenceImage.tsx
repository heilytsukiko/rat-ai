import { useState } from 'react';
import { ImageOff } from 'lucide-react';

export function EvidenceImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [failed, setFailed] = useState(false);
  return failed ? <span className="image-error" role="status"><ImageOff size={26} /><strong>Кадр недоступен</strong><span>Не удалось загрузить изображение.</span></span>
    : <img src={src} alt={alt} className={className} onError={() => setFailed(true)} />;
}
