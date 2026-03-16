export interface ProgressEvent {
  documentId: string;
  modelLabel: string;
  runNumber: number;
  ok: boolean;
  error?: string | null;
}

export function createProgressReporter(params: {
  totalTasks: number;
  expectedRunsByDocument: Map<string, number>;
  startedAtMs: number;
}) {
  const { totalTasks, expectedRunsByDocument, startedAtMs } = params;

  const completedByDocument = new Map<string, number>();
  let completedTasks = 0;
  let successfulTasks = 0;
  let failedTasks = 0;
  const logEvery = Math.max(1, Math.floor(totalTasks / 20));

  return (event: ProgressEvent) => {
    completedTasks += 1;
    if (event.ok) {
      successfulTasks += 1;
    } else {
      failedTasks += 1;
    }

    const docCompleted = (completedByDocument.get(event.documentId) ?? 0) + 1;
    completedByDocument.set(event.documentId, docCompleted);

    const percent = totalTasks > 0 ? ((completedTasks / totalTasks) * 100).toFixed(1) : '100.0';
    const elapsedSec = ((Date.now() - startedAtMs) / 1000).toFixed(1);
    const expectedForDocument = expectedRunsByDocument.get(event.documentId) ?? 0;
    const documentDone = expectedForDocument > 0 && docCompleted === expectedForDocument;
    const periodicTick = completedTasks === 1 || completedTasks === totalTasks || completedTasks % logEvery === 0;

    if (periodicTick || documentDone || !event.ok) {
      const docSuffix = documentDone
        ? ` | document complete: ${event.documentId} (${docCompleted}/${expectedForDocument})`
        : '';
      const errorSuffix = !event.ok ? ` | last error: ${event.error ?? 'unknown error'}` : '';
      console.log(
        `[ocr-benchmark] ${completedTasks}/${totalTasks} (${percent}%) | ok:${successfulTasks} fail:${failedTasks} | elapsed:${elapsedSec}s | model:${event.modelLabel} run:${event.runNumber}${docSuffix}${errorSuffix}`
      );
    }
  };
}
