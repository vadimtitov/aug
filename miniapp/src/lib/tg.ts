// The Telegram WebApp object is injected on `window` at runtime by Telegram's
// SDK script; it has no ambient type here, so the cast is intentional.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const tg = () => (window as any).Telegram?.WebApp;

export const isInTelegram = () => Boolean(tg()?.initData);

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
