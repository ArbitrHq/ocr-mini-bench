export async function runInPool<T>(
  items: T[],
  maxWorkers: number,
  runner: (item: T) => Promise<void>
): Promise<void> {
  const workerCount = Math.max(1, Math.min(maxWorkers, items.length));
  let cursor = 0;

  const workers = Array.from({ length: workerCount }).map(async () => {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) {
        return;
      }
      await runner(items[index]);
    }
  });

  await Promise.all(workers);
}
