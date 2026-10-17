import { ShieldCheck } from 'lucide-react';

export function AppHeader() {
  return <header className="app-header">
    <a className="brand" href="#main" aria-label="Forte — контроль кассовых операций">
      <span className="brand-mark" aria-hidden="true">f</span><span className="brand-name">forte</span>
      <span className="brand-caption">Контроль кассовых операций</span>
    </a>
    <div className="header-meta"><ShieldCheck size={15} /><span>Демонстрационный режим</span></div>
  </header>;
}
