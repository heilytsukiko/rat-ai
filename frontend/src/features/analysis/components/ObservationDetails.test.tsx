import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { EventDetails } from './EventDetails';

it('renders the actual report instead of the disconnected-model placeholder', () => {
  render(<EventDetails index={0} total={1} experimental event={{ id: 'test', type: 'Разбор эпизода',
    timecodeSec: 2, description: 'Прямоугольник перемещается.', evidenceFrames: [],
    detector: { score: .6, threshold: .5, hasObservation: true },
    observationReport: { summary: 'Прямоугольник перемещается.',
      actions: [{ timestamp_sec: 2, description: 'Прямоугольник справа.' }],
      object_description: 'Красный прямоугольник.', assessment: 'no_transfer_visible',
      reason: 'Людей нет.', limitations: ['Между кадрами могут быть пропуски.'],
      sampled_timestamps_sec: [0, 1, 2, 3, 4, 5, 6, 7] } }} />);
  expect(screen.getByRole('heading', { name: 'Последовательность действий' })).toBeInTheDocument();
  expect(screen.getByText('00:02.0')).toBeInTheDocument();
  expect(screen.getByText('Красный прямоугольник.')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Передача предмета не видна' })).toBeInTheDocument();
  expect(screen.queryByText('Почему нет описания действия?')).not.toBeInTheDocument();
});
