import type { ObservationReport } from '../model/types';
import { formatTime } from '../model/format';

const assessments: Record<ObservationReport['assessment'], string> = {
  possible_improper_transfer: 'Признаки возможной неслужебной передачи',
  ordinary_exchange: 'Обмен без выявленных признаков неслужебной передачи',
  no_transfer_visible: 'Передача предмета не видна',
  unclear: 'Назначение передачи определить не удалось',
};

export function ObservationDetails({ report }: { report: ObservationReport }) {
  return <div className="observation-details">
    <section className="regulatory-source"><h4>Последовательность действий</h4>
      <ol>{report.actions.map((action, index) => <li key={`${action.timestamp_sec}-${index}`}>
        <time>{formatTime(action.timestamp_sec, true)}</time> — {action.description}
      </li>)}</ol>
    </section>
    <section className="regulatory-source"><h4>Передаваемый предмет</h4><p>{report.object_description}</p></section>
    <section className="regulatory-source"><h4>{assessments[report.assessment]}</h4><p>{report.reason}</p></section>
    <section className="regulatory-source"><h4>Что остаётся установить</h4>
      <ul>{report.limitations.map((item, index) => <li key={index}>{item}</li>)}</ul>
      <p>Вывод модели предварительный. Сам факт передачи не устанавливает взятку: нужно проверить назначение передачи и связь с действиями получателя. Видео анализируется по выбранным кадрам, отдельные движения могут быть пропущены.</p>
    </section>
  </div>;
}
