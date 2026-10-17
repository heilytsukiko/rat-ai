import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AnalysisResults } from './AnalysisResults';
import { createMockResult } from '../mocks/results';
import { EvidenceImage } from './EvidenceImage';
import { ProcessingSteps } from './ProcessingSteps';
import { FileDropzone } from './FileDropzone';

describe('review and upload regressions', () => {
  it('does not present an empty experimental result as absence of violations', () => {
    const result = { ...createMockResult('no_violation'), isDemo: false,
      experimental: true, verdict: 'review_required' as const };
    render(<AnalysisResults state={{ phase: 'done', result }} onRetry={() => {}} />);
    expect(screen.getByRole('heading', { name: 'Детектор не выделил фрагменты' })).toBeInTheDocument();
    expect(screen.getByText('Экспериментальный анализ — требуется проверка')).toBeInTheDocument();
    expect(screen.queryByText('Нарушение не обнаружено')).not.toBeInTheDocument();
  });
  it('labels a preview explicitly and navigates events without wrapping', () => {
    render(<AnalysisResults state={{ phase: 'preview', result: createMockResult('violation') }} onRetry={() => {}} />);
    expect(screen.getByText('Образец результата', { exact: false })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Предыдущее событие' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Следующее событие' }));
    expect(screen.getByRole('heading', { name: 'Передача предмета через стойку' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Следующее событие' })).toBeDisabled();
  });
  it('shows explicit cancellation instead of an unexplained empty result', () => {
    render(<AnalysisResults state={{ phase: 'cancelled' }} onRetry={() => {}} />);
    expect(screen.getByRole('heading', { name: 'Ожидание прекращено' })).toBeInTheDocument();
  });
  it('provides a fallback for a missing frame', () => {
    render(<EvidenceImage src="missing.jpg" alt="Кадр события" />);
    fireEvent.error(screen.getByRole('img'));
    expect(screen.getByRole('status')).toHaveTextContent('Кадр недоступен');
  });
  it('forwards all dropped files and blocks drops while busy', () => {
    const select = vi.fn();
    const files = [new File(['a'], 'a.mov'), new File(['b'], 'b.mov')];
    const { container, rerender } = render(<FileDropzone onSelect={select} disabled={false} invalid={false} />);
    fireEvent.drop(container.firstChild!, { dataTransfer: { files } });
    expect(select).toHaveBeenCalledWith(files);
    select.mockClear();
    rerender(<FileDropzone onSelect={select} disabled invalid={false} />);
    fireEvent.drop(container.firstChild!, { dataTransfer: { files } });
    expect(select).not.toHaveBeenCalled();
  });
});

it('keeps completed upload marked done when polling is stopped', () => {
  render(<ProcessingSteps state={{ phase: 'cancelled', stage: 'processing' }} />);
  const rows = screen.getAllByRole('listitem');
  expect(rows[0]).toHaveTextContent('Готово');
  expect(rows[1]).toHaveTextContent('Нет статуса');
  expect(rows[2]).toHaveTextContent('Ожидание');
});
