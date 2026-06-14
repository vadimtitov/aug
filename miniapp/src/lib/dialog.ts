/**
 * In-app confirm/alert dialog controller.
 *
 * Telegram's `showConfirm`/`showAlert` popups are only available from Bot API
 * 6.2 and are missing on some clients; the browser's native `confirm()`/
 * `alert()` are unreliable inside Telegram's mobile WebView. Neither is
 * dependable everywhere, so we render our own modal instead — it works
 * identically in a plain browser, Telegram Web, Desktop, iOS and Android.
 *
 * The imperative `confirmDialog()` / `alertDialog()` helpers can be called from
 * anywhere (including non-React code). They drive a single <DialogHost/> (see
 * dialog-host.tsx) which must be mounted once near the app root.
 */

export interface ConfirmOptions {
  confirmText?: string;
  cancelText?: string;
  destructive?: boolean;
}

export type DialogRequest =
  | { kind: "confirm"; message: string; options: ConfirmOptions; resolve: (ok: boolean) => void }
  | { kind: "alert"; message: string; resolve: () => void };

let _enqueue: ((req: DialogRequest) => void) | null = null;

/** Called by <DialogHost/> on mount. Returns an unsubscribe for unmount. */
export function registerDialogHost(enqueue: (req: DialogRequest) => void): () => void {
  _enqueue = enqueue;
  return () => {
    if (_enqueue === enqueue) _enqueue = null;
  };
}

/** Ask the user to confirm. Resolves true if they accept, false otherwise. */
export function confirmDialog(message: string, options: ConfirmOptions = {}): Promise<boolean> {
  return new Promise((resolve) => {
    if (_enqueue) {
      _enqueue({ kind: "confirm", message, options, resolve });
    } else {
      // Host not mounted (shouldn't happen in practice) — degrade gracefully.
      resolve(window.confirm(message));
    }
  });
}

/** Show a message the user must dismiss. Resolves once dismissed. */
export function alertDialog(message: string): Promise<void> {
  return new Promise((resolve) => {
    if (_enqueue) {
      _enqueue({ kind: "alert", message, resolve });
    } else {
      window.alert(message);
      resolve();
    }
  });
}
