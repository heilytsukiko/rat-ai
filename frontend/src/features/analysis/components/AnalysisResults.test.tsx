import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AnalysisResults } from './AnalysisResults';
import { createMockResult } from '../mocks/results';
import { EvidenceImage } from './EvidenceImage';
import { FileDropzone } from './FileDropzone';

describe('review and upload regressions', () => {
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
    expect(screen.getByRole('heading', { name: 'Проверка отменена' })).toBeInTheDocument();
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
