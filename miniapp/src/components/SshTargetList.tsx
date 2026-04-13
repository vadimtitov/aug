/**
 * Editor for SSH targets — list of { name, host, user, key_path, port? } objects.
 */

import { useState } from "react";
import { X, Plus } from "lucide-react";

export interface SshTarget {
  name: string;
  host: string;
  user: string;
  key_path: string;
  port?: number;
}

interface SshTargetListProps {
  targets: SshTarget[];
  onChange: (targets: SshTarget[]) => void;
}

const _empty = (): SshTarget => ({ name: "", host: "", user: "", key_path: "" });

export function SshTargetList({ targets, onChange }: SshTargetListProps) {
  const [draft, setDraft] = useState<SshTarget>(_empty());
  const [adding, setAdding] = useState(false);

  function set(field: keyof SshTarget, val: string) {
    setDraft((d) => ({ ...d, [field]: field === "port" ? Number(val) || undefined : val }));
  }

  function add() {
    if (!draft.name || !draft.host || !draft.user || !draft.key_path) return;
    onChange([...targets, draft]);
    setDraft(_empty());
    setAdding(false);
  }

  function cancel() {
    setDraft(_empty());
    setAdding(false);
  }

  function remove(name: string) {
    onChange(targets.filter((t) => t.name !== name));
  }

  const canAdd = !!(draft.name && draft.host && draft.user && draft.key_path);

  return (
    <div className="settings-section">
      <div className="settings-section-label">SSH Targets</div>
      <div className="settings-card">
        {targets.map((t) => (
          <div key={t.name} className="list-item">
            <div className="list-item-main">
              <div className="list-item-title">{t.name}</div>
              <div className="list-item-sub">
                {t.user}@{t.host}{t.port ? `:${t.port}` : ""} &mdash; {t.key_path}
              </div>
            </div>
            <button className="icon-btn" onClick={() => remove(t.name)} aria-label="Remove">
              <X size={14} strokeWidth={2.5} />
            </button>
          </div>
        ))}

        {adding ? (
          <div className="expand-form">
            {(["name", "host", "user", "key_path"] as const).map((field) => (
              <div key={field} className="expand-form-field">
                <label className="expand-form-label">
                  {field === "key_path" ? "Key path" : field.charAt(0).toUpperCase() + field.slice(1)}
                </label>
                <input
                  className="expand-form-input"
                  placeholder={field === "key_path" ? "/home/user/.ssh/id_rsa" : undefined}
                  value={(draft[field] as string) ?? ""}
                  onChange={(e) => set(field, e.target.value)}
                />
              </div>
            ))}
            <div className="expand-form-field">
              <label className="expand-form-label">Port (optional)</label>
              <input
                className="expand-form-input"
                type="number"
                placeholder="22"
                value={draft.port?.toString() ?? ""}
                onChange={(e) => set("port", e.target.value)}
              />
            </div>
            <div className="expand-form-actions">
              <button className="btn-secondary" onClick={cancel}>Cancel</button>
              <button className="btn-primary" disabled={!canAdd} onClick={add}>Save</button>
            </div>
          </div>
        ) : (
          <button className="action-row" onClick={() => setAdding(true)}>
            <Plus size={16} strokeWidth={2.5} />
            Add SSH target
          </button>
        )}
      </div>
    </div>
  );
}
