import { ShieldCheck } from 'lucide-react';

export function AppHeader() {
  return <header className="app-header">
    <a className="brand" href="#main" aria-label="Rat AI — контроль кассовых операций">
      <span className="brand-mark" aria-hidden="true">AI</span><span className="brand-name">Rat</span>
      <span className="brand-caption">Контроль кассовых операций</span>
    </a>
    <div className="header-meta"><ShieldCheck size={15} /><span>Демонстрационный режим</span></div>
  </header>;
}
