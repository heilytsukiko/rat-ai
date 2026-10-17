import { FlaskConical, Play } from 'lucide-react';
import type { DemoScenario } from '../model/types';

interface DemoControlsProps {
  scenario: DemoScenario;
  onScenarioChange: (value: DemoScenario) => void;
  onRun: () => void;
  disabled: boolean;
}

export function DemoControls({ scenario, onScenarioChange, onRun, disabled }: DemoControlsProps) {
  return <section className="demo-controls" aria-label="Настройки демонстрации">
    <div className="demo-explanation"><FlaskConical size={17} /><div><strong>Попробуйте на примере</strong><span>Условные события и схемы кадров. Реальный анализ пока не подключён.</span></div></div>
    <div className="demo-actions">
      <label className="sr-only" htmlFor="scenario">Демонстрационный сценарий</label>
      <select id="scenario" value={scenario} disabled={disabled} onChange={event => onScenarioChange(event.target.value as DemoScenario)}>
        <option value="violation">Есть события</option><option value="no_violation">Без нарушений</option><option value="failed">Ошибка обработки</option>
      </select>
      <button className="button button-secondary" disabled={disabled} onClick={onRun}><Play size={14} />Запустить пример</button>
    </div>
  </section>;
}
