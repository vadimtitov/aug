let _version = 0;
const _listeners = new Set<() => void>();

export function getInstalledVersion(): number {
  return _version;
}

export function bumpInstalledVersion(): void {
  _version++;
  _listeners.forEach((fn) => fn());
}

export function onInstalledVersionChange(fn: () => void): () => void {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}
