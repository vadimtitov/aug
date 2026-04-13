/**
 * Dropdown that lists available models from GET /models.
 * Shows the current value while loading and falls back gracefully on error.
 */

import { useEffect, useState } from "react";
import { getModels } from "../api.ts";

interface ModelSelectProps {
  label: string;
  value: string;
  onChange: (model: string) => void;
}

export function ModelSelect({ label, value, onChange }: ModelSelectProps) {
  const [models, setModels] = useState<string[]>([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch(() => setError(true));
  }, []);

  return (
    <div className="settings-row">
      <span className="settings-row-label">{label}</span>
      <select
        className="field-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={error}
      >
        {error && <option value={value}>{value || "Unavailable"}</option>}
        {!error && value && !models.includes(value) && (
          <option value={value}>{value}</option>
        )}
        {!error && !value && <option value="">Select…</option>}
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
