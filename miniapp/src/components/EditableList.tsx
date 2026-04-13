/**
 * Editable list of strings — used for bash blocklist, etc.
 */

import { useState } from "react";
import { X, Plus } from "lucide-react";

interface EditableListProps {
  header: string;
  placeholder?: string;
  items: string[];
  onChange: (items: string[]) => void;
}

export function EditableList({
  header,
  placeholder = "Add item…",
  items,
  onChange,
}: EditableListProps) {
  const [draft, setDraft] = useState("");

  function add() {
    const trimmed = draft.trim();
    if (!trimmed || items.includes(trimmed)) return;
    onChange([...items, trimmed]);
    setDraft("");
  }

  function remove(item: string) {
    onChange(items.filter((i) => i !== item));
  }

  return (
    <div className="settings-section">
      <div className="settings-section-label">{header}</div>
      <div className="settings-card">
        {items.map((item) => (
          <div key={item} className="list-item">
            <div className="list-item-main">
              <div className="list-item-title">{item}</div>
            </div>
            <button className="icon-btn" onClick={() => remove(item)} aria-label="Remove">
              <X size={14} strokeWidth={2.5} />
            </button>
          </div>
        ))}
        <div className="add-row">
          <Plus size={16} color="var(--hint)" strokeWidth={2} />
          <input
            className="add-row-input"
            placeholder={placeholder}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
          />
          <button className="add-btn" disabled={!draft.trim()} onClick={add}>
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
