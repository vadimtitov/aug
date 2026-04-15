export const tg = () => (window as any).Telegram?.WebApp;

export const isInTelegram = () => Boolean(tg()?.initData);

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
