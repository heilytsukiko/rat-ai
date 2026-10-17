import cashFrame from '../../../assets/evidence-cash.svg';
import transferFrame from '../../../assets/evidence-transfer.svg';
import type { AnalysisResult, DemoScenario } from '../model/types';

export const DEMO_FILE_NAME = 'Пример кассовой операции.mov';

export function createMockResult(scenario: Exclude<DemoScenario, 'failed'>, fileName = DEMO_FILE_NAME): AnalysisResult {
  return {
    jobId: `demo-${scenario}`,
    fileName,
    // Duration describes only the bundled example, never the user's file.
    durationSec: fileName === DEMO_FILE_NAME ? 42 : undefined,
    isDemo: true,
    verdict: scenario,
    violations: scenario === 'no_violation' ? [] : [
      {
        id: 'cash-outside-zone',
        type: 'Наличные вне рабочей зоны',
        timecodeSec: 11,
        description: 'В условном примере банкноты находятся у края рабочей поверхности, вне зоны пересчёта. Проверяющему необходимо просмотреть фрагмент и сопоставить действие с применимым правилом.',
        evidenceFrames: [{
          id: 'cash-frame', timestampSec: 11, imageUrl: cashFrame,
          alt: 'Демонстрационная схема кассы: банкноты у края рабочей поверхности',
        }],
      },
      {
        id: 'object-transfer',
        type: 'Передача предмета через стойку',
        timecodeSec: 27.4,
        description: 'В условном примере показана передача предмета другому человеку через стойку. По одному кадру нельзя установить назначение передачи: требуется проверка последовательности действий.',
        evidenceFrames: [{
          id: 'transfer-frame', timestampSec: 27.4, imageUrl: transferFrame,
          alt: 'Демонстрационная схема кассы: выделена зона передачи предмета',
        }],
      },
    ],
  };
}
