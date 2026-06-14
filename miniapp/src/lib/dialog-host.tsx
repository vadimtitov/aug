/**
 * Renders the active in-app dialog (see dialog.ts). Mount once near the app
 * root; the imperative confirmDialog()/alertDialog() helpers drive it.
 */

import { useEffect, useState } from "react";
import { registerDialogHost, type DialogRequest } from "./dialog.ts";

export function DialogHost() {
  const [req, setReq] = useState<DialogRequest | null>(null);

  useEffect(() => registerDialogHost(setReq), []);

  if (!req) return null;

  function close(result: boolean) {
    if (!req) return;
    if (req.kind === "confirm") req.resolve(result);
    else req.resolve();
    setReq(null);
  }

  const confirmText = req.kind === "confirm" ? req.options.confirmText ?? "Confirm" : "OK";
  const destructive = req.kind === "confirm" && req.options.destructive;

  return (
    <div className="modal-overlay" onClick={() => close(false)}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <p className="modal-body" style={{ margin: "0 0 20px", color: "var(--text)" }}>
          {req.message}
        </p>
        <div className="modal-actions">
          {req.kind === "confirm" && (
            <button className="btn-secondary" onClick={() => close(false)}>
              {req.options.cancelText ?? "Cancel"}
            </button>
          )}
          <button
            className={destructive ? "btn-primary btn-primary--warn" : "btn-primary"}
            onClick={() => close(true)}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
