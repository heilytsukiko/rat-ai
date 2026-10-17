import { AppFooter } from '../../components/AppFooter';
import { AppHeader } from '../../components/AppHeader';
import { SectionLabel } from '../../components/ui/SectionLabel';
import { AnalysisResults } from './components/AnalysisResults';
import { DemoControls } from './components/DemoControls';
import { UploadPanel } from './components/UploadPanel';
import { useAnalysis } from './hooks/useAnalysis';

export function AnalysisPage() {
  const analysis = useAnalysis();
  return <>
    <a className="skip-link" href="#main">Перейти к содержимому</a>
    <AppHeader />
    <main className="page-shell" id="main">
      <section className="page-intro"><div><SectionLabel>ForteBank / Видеоконтроль</SectionLabel><h1>Проверка кассовых операций</h1><p>Загрузите запись и изучите события, требующие внимания.</p></div><span className="version-badge">Прототип <span>01</span></span></section>
      <DemoControls scenario={analysis.scenario} onScenarioChange={analysis.setScenario} onRun={() => void analysis.start(true)} disabled={analysis.busy} />
      <div className="workspace">
        <UploadPanel file={analysis.file} error={analysis.fileError} state={analysis.state} busy={analysis.busy} onSelect={analysis.selectFiles} onClear={analysis.clearFile} onStart={() => void analysis.start()} onCancel={analysis.cancel} />
        <AnalysisResults state={analysis.state} onRetry={() => void analysis.retry()} />
      </div>
      <AppFooter />
    </main>
  </>;
}
