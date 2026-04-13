/**
 * Editor for tool approval rules — list of { tool, target, pattern } objects.
 * Rules use re.search against the operation string; "*" is a wildcard for tool/target.
 */

import { useState } from "react";
import { X, Plus } from "lucide-react";

export interface ApprovalRule {
  tool: string;
  target: string;
  pattern: string;
}

interface ApprovalRuleListProps {
  rules: ApprovalRule[];
  onChange: (rules: ApprovalRule[]) => void;
}

const _empty = (): ApprovalRule => ({ tool: "", target: "*", pattern: ".*" });

export function ApprovalRuleList({ rules, onChange }: ApprovalRuleListProps) {
  const [draft, setDraft] = useState<ApprovalRule>(_empty());
  const [adding, setAdding] = useState(false);

  function set(field: keyof ApprovalRule, val: string) {
    setDraft((d) => ({ ...d, [field]: val }));
  }

  function add() {
    if (!draft.tool || !draft.pattern) return;
    onChange([...rules, draft]);
    setDraft(_empty());
    setAdding(false);
  }

  function cancel() {
    setDraft(_empty());
    setAdding(false);
  }

  function remove(index: number) {
    onChange(rules.filter((_, i) => i !== index));
  }

  return (
    <div className="settings-section">
      <div className="settings-section-label">Approval Rules</div>
      <div className="settings-card">
        {rules.map((rule, i) => (
          <div key={i} className="list-item">
            <div className="list-item-main">
              <div className="list-item-title">{rule.tool}</div>
              <div className="list-item-sub">
                target: {rule.target} &mdash; pattern: {rule.pattern}
              </div>
            </div>
            <button className="icon-btn" onClick={() => remove(i)} aria-label="Remove">
              <X size={14} strokeWidth={2.5} />
            </button>
          </div>
        ))}

        {adding ? (
          <div className="expand-form">
            <div className="expand-form-field">
              <label className="expand-form-label">Tool name</label>
              <input
                className="expand-form-input"
                placeholder="e.g. run_ssh, * for any"
                value={draft.tool}
                onChange={(e) => set("tool", e.target.value)}
              />
            </div>
            <div className="expand-form-field">
              <label className="expand-form-label">Target</label>
              <input
                className="expand-form-input"
                placeholder="* for any"
                value={draft.target}
                onChange={(e) => set("target", e.target.value)}
              />
            </div>
            <div className="expand-form-field">
              <label className="expand-form-label">Pattern (regex)</label>
              <input
                className="expand-form-input"
                placeholder=".*"
                value={draft.pattern}
                onChange={(e) => set("pattern", e.target.value)}
              />
            </div>
            <div className="expand-form-actions">
              <button className="btn-secondary" onClick={cancel}>Cancel</button>
              <button className="btn-primary" disabled={!draft.tool || !draft.pattern} onClick={add}>
                Save
              </button>
            </div>
          </div>
        ) : (
          <button className="action-row" onClick={() => setAdding(true)}>
            <Plus size={16} strokeWidth={2.5} />
            Add approval rule
          </button>
        )}
      </div>
    </div>
  );
}
