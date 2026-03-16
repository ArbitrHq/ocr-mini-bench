import { runInPool } from './pool';

export async function runByProviderLanes<T extends { provider: string }>(
  tasks: T[],
  maxProviderLanes: number,
  runner: (task: T) => Promise<void>
): Promise<void> {
  const byProvider = new Map<string, T[]>();

  for (const task of tasks) {
    const list = byProvider.get(task.provider) ?? [];
    list.push(task);
    byProvider.set(task.provider, list);
  }

  const providerQueues = Array.from(byProvider.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([provider, providerTasks]) => ({ provider, tasks: providerTasks }));

  await runInPool(providerQueues, maxProviderLanes, async (queue) => {
    for (const task of queue.tasks) {
      await runner(task);
    }
  });
}
