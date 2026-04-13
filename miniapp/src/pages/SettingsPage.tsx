/**
 * Main settings page.
 *
 * Loads the full settings object on mount, renders each section, and saves
 * the entire object back on any change (debounced 800ms to avoid hammering
 * the API on rapid edits).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft } from "lucide-react";
import { getSettings, putSettings } from "../api.ts";
import { ModelSelect } from "../components/ModelSelect.tsx";
import { EditableList } from "../components/EditableList.tsx";
import { SshTargetList, type SshTarget } from "../components/SshTargetList.tsx";
import { ApprovalRuleList, type ApprovalRule } from "../components/ApprovalRuleList.tsx";

// ---------------------------------------------------------------------------
// Helpers to safely read/write deeply nested paths
// ---------------------------------------------------------------------------

function getPath(obj: Record<string, unknown>, ...path: string[]): unknown {
  let node: unknown = obj;
  for (const key of path) {
    if (node == null || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[key];
  }
  return node;
}

function setPath(
  obj: Record<string, unknown>,
  path: string[],
  value: unknown
): Record<string, unknown> {
  const result = { ...obj };
  let node = result;
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i];
    node[key] = typeof node[key] === "object" && node[key] !== null
      ? { ...(node[key] as object) }
      : {};
    node = node[key] as Record<string, unknown>;
  }
  node[path[path.length - 1]] = value;
  return result;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SettingsPageProps {
  onBack: () => void;
}

export function SettingsPage({ onBack }: SettingsPageProps) {
  const [settings, setSettings] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSettings()
      .then((s) => { if (!cancelled) setSettings(s); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, []);

  const update = useCallback(
    (path: string[], value: unknown) => {
      setSettings((prev) => {
        if (!prev) return prev;
        const next = setPath(prev, path, value);

        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(() => {
          setSaving(true);
          putSettings(next)
            .catch((e: Error) => setError(e.message))
            .finally(() => setSaving(false));
        }, 800);

        return next;
      });
    },
    []
  );

  if (error) {
    return (
      <div className="screen">
        <div className="page-header">
          <button className="back-btn" onClick={onBack}>
            <ChevronLeft size={20} strokeWidth={2.5} />
            Back
          </button>
        </div>
        <div className="centered">
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="screen">
        <div className="centered">
          <div className="spinner" />
        </div>
      </div>
    );
  }

  const browserModel = (getPath(settings, "tools", "browser", "model") as string) ?? "";
  const consolidationModel = (getPath(settings, "consolidation", "model") as string) ?? "";
  const imageGenModel = (getPath(settings, "tools", "image_gen", "model") as string) ?? "";
  const bashBlacklist = (getPath(settings, "tools", "bash", "blacklist") as string[]) ?? [];
  const sshTargets = (getPath(settings, "tools", "ssh", "targets") as SshTarget[]) ?? [];
  const haLabel = (getPath(settings, "reflexes", "homeassistant", "entity_label") as string) ?? "";
  const approvalRules = (getPath(settings, "tools", "approvals") as ApprovalRule[]) ?? [];
  const sshMaxBytes = (getPath(settings, "tools", "ssh", "max_download_bytes") as number) ?? "";

  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} strokeWidth={2.5} />
          Back
        </button>
        <h1>Settings</h1>
        {saving && <div className="save-dot" />}
      </div>

      <div className="settings-content">

        {/* Models */}
        <div className="settings-section">
          <div className="settings-section-label">Models</div>
          <div className="settings-card">
            <ModelSelect
              label="Browser tool"
              value={browserModel}
              onChange={(v) => update(["tools", "browser", "model"], v)}
            />
            <ModelSelect
              label="Consolidation"
              value={consolidationModel}
              onChange={(v) => update(["consolidation", "model"], v)}
            />
            <ModelSelect
              label="Image generation"
              value={imageGenModel}
              onChange={(v) => update(["tools", "image_gen", "model"], v)}
            />
          </div>
        </div>

        {/* Home Assistant */}
        <div className="settings-section">
          <div className="settings-section-label">Home Assistant</div>
          <div className="settings-card">
            <div className="settings-row">
              <span className="settings-row-label">Entity label filter</span>
              <input
                className="field-input"
                placeholder="aug"
                value={haLabel}
                onChange={(e) =>
                  update(["reflexes", "homeassistant", "entity_label"], e.target.value)
                }
              />
            </div>
          </div>
        </div>

        {/* SSH file transfer */}
        <div className="settings-section">
          <div className="settings-section-label">SSH File Transfer</div>
          <div className="settings-card">
            <div className="settings-row">
              <span className="settings-row-label">Max bytes per transfer</span>
              <input
                className="field-input"
                type="number"
                placeholder="1048576"
                value={sshMaxBytes.toString()}
                onChange={(e) =>
                  update(["tools", "ssh", "max_download_bytes"], Number(e.target.value) || undefined)
                }
              />
            </div>
          </div>
        </div>

        {/* Bash blocklist */}
        <EditableList
          header="Bash Blocklist"
          placeholder="Regex pattern (e.g. rm\s+-rf)"
          items={bashBlacklist}
          onChange={(v) => update(["tools", "bash", "blacklist"], v)}
        />

        {/* SSH targets */}
        <SshTargetList
          targets={sshTargets}
          onChange={(v) => update(["tools", "ssh", "targets"], v)}
        />

        {/* Approval rules */}
        <ApprovalRuleList
          rules={approvalRules}
          onChange={(v) => update(["tools", "approvals"], v)}
        />

      </div>
    </div>
  );
}
