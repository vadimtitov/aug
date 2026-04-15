import { useEffect, useState } from "react";
import {
  ChevronLeft, Pencil, Check, X, File,
  AlertTriangle, Download, RefreshCw, Star, Users,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  clawhubGetFile,
  clawhubGetSkill,
  deleteSkill,
  getSkillDetail,
  installSkill,
  listSkills,
  updateSkill,
} from "../api.ts";
import type { ClawHubSkillDetail as ClawHubDetail, PageState, SkillDetail } from "../types.ts";

interface Props {
  skillName: string;
  source: "local" | "clawhub";
  slug?: string;
  onBack: () => void;
  onNavigate: (state: PageState) => void;
  onDeleted: () => void;
}

export function SkillDetailPage({ skillName, source, slug, onBack, onNavigate, onDeleted }: Props) {
  if (source === "local") {
    return (
      <LocalSkillDetail
        skillName={skillName}
        onBack={onBack}
        onNavigate={onNavigate}
        onDeleted={onDeleted}
      />
    );
  }
  return (
    <ClawHubSkillDetailPage
      slug={slug!}
      onBack={onBack}
      onNavigate={onNavigate}
    />
  );
}

// ---------------------------------------------------------------------------
// Local skill detail
// ---------------------------------------------------------------------------

function LocalSkillDetail({
  skillName,
  onBack,
  onNavigate,
  onDeleted,
}: {
  skillName: string;
  onBack: () => void;
  onNavigate: (s: PageState) => void;
  onDeleted: () => void;
}) {
  const [skill, setSkill] = useState<SkillDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [editing, setEditing] = useState(false);
  const [editBody, setEditBody] = useState("");
  const [saving, setSaving] = useState(false);

  const [desc, setDesc] = useState("");
  const [savingDesc, setSavingDesc] = useState(false);

  const [alwaysOn, setAlwaysOn] = useState(false);
  const [savingToggle, setSavingToggle] = useState(false);

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    getSkillDetail(skillName)
      .then((s) => {
        setSkill(s);
        setDesc(s.description);
        setAlwaysOn(s.always_on);
        setEditBody(s.body);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [skillName]);

  function startEdit() {
    setEditBody(skill!.body);
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setEditBody(skill!.body);
  }

  async function saveBody() {
    if (!skill) return;
    setSaving(true);
    try {
      await updateSkill(skillName, { body: editBody });
      setSkill({ ...skill, body: editBody });
      setEditing(false);
    } catch (e) {
      alert(`Failed to save: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  async function saveDesc() {
    if (!skill || desc === skill.description) return;
    setSavingDesc(true);
    try {
      await updateSkill(skillName, { description: desc });
      setSkill({ ...skill, description: desc });
    } catch {
      setDesc(skill.description);
    } finally {
      setSavingDesc(false);
    }
  }

  async function toggleAlwaysOn() {
    if (!skill || savingToggle) return;
    const next = !alwaysOn;
    setAlwaysOn(next);
    setSavingToggle(true);
    try {
      await updateSkill(skillName, { always_on: next });
      setSkill({ ...skill, always_on: next });
    } catch {
      setAlwaysOn(!next);
    } finally {
      setSavingToggle(false);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteSkill(skillName);
      onDeleted();
    } catch (e) {
      alert(`Failed to delete: ${e}`);
      setDeleting(false);
      setConfirmDelete(false);
    }
  }

  if (loading) return <LoadingScreen onBack={onBack} title={skillName} />;
  if (error || !skill) return <ErrorScreen onBack={onBack} message={error ?? "Not found"} />;

  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1 style={{ fontSize: 20 }}>{skill.name}</h1>
        {!editing ? (
          <button className="icon-action-btn" onClick={startEdit}>
            <Pencil size={18} color="var(--brand)" />
          </button>
        ) : (
          <div style={{ display: "flex", gap: 8 }}>
            <button className="icon-action-btn" onClick={cancelEdit} disabled={saving}>
              <X size={18} color="var(--hint)" />
            </button>
            <button className="icon-action-btn" onClick={saveBody} disabled={saving}>
              <Check size={18} color="var(--brand)" />
            </button>
          </div>
        )}
      </div>

      <div className="detail-content">
        <div className="settings-section">
          <div className="settings-section-label">Description</div>
          <div className="settings-card">
            <div className="settings-row">
              <input
                className="field-input"
                style={{ textAlign: "left" }}
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                onBlur={saveDesc}
                disabled={savingDesc}
              />
            </div>
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-card">
            <div className="settings-row">
              <span className="settings-row-label">Always active</span>
              <span style={{ fontSize: 12, color: "var(--hint)", marginRight: 8 }}>
                inject into every prompt
              </span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={alwaysOn}
                  onChange={toggleAlwaysOn}
                  disabled={savingToggle}
                />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-section-label">Instructions</div>
          {editing ? (
            <textarea
              className="edit-textarea"
              value={editBody}
              onChange={(e) => setEditBody(e.target.value)}
              rows={16}
            />
          ) : (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{skill.body}</ReactMarkdown>
            </div>
          )}
        </div>

        {skill.files.length > 0 && (
          <div className="settings-section">
            <div className="settings-section-label">Files</div>
            <div className="settings-card">
              {skill.files.map((f) => (
                <button
                  key={f}
                  className="file-row"
                  onClick={() =>
                    onNavigate({
                      page: "file-viewer",
                      skillName: skill.name,
                      filePath: f,
                      source: "local",
                    })
                  }
                >
                  <File size={15} color="var(--hint)" />
                  <span className="file-row-name">{f}</span>
                  <ChevronLeft size={14} color="var(--hint)" style={{ transform: "rotate(180deg)" }} />
                </button>
              ))}
            </div>
          </div>
        )}

        <div style={{ padding: "8px 0 40px" }}>
          <button className="btn-danger" onClick={() => setConfirmDelete(true)}>
            Delete Skill
          </button>
        </div>
      </div>

      {confirmDelete && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">Delete "{skill.name}"?</h3>
            <p className="modal-body">This cannot be undone.</p>
            <div className="modal-actions">
              <button
                className="btn-secondary"
                onClick={() => setConfirmDelete(false)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button className="btn-danger" onClick={handleDelete} disabled={deleting}>
                {deleting ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ClawHub skill detail
// ---------------------------------------------------------------------------

function ClawHubSkillDetailPage({
  slug,
  onBack,
  onNavigate,
}: {
  slug: string;
  onBack: () => void;
  onNavigate: (s: PageState) => void;
}) {
  const [detail, setDetail] = useState<ClawHubDetail | null>(null);
  const [body, setBody] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installed, setInstalled] = useState(false);
  const [confirmOverwrite, setConfirmOverwrite] = useState(false);

  useEffect(() => {
    Promise.all([
      clawhubGetSkill(slug),
      clawhubGetFile(slug, "SKILL.md").catch(() => null),
    ])
      .then(([d, b]) => {
        setDetail(d);
        setBody(b);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [slug]);

  useEffect(() => {
    listSkills()
      .then((skills) => setInstalled(skills.some((s) => s.name === slug)))
      .catch(() => {});
  }, [slug]);

  async function doInstall() {
    if (!detail) return;
    setInstalling(true);
    setConfirmOverwrite(false);
    try {
      await installSkill(slug, slug, detail.latestVersion.version);
      setInstalled(true);
    } catch (e) {
      alert(`Install failed: ${e}`);
    } finally {
      setInstalling(false);
    }
  }

  function handleInstallClick() {
    if (installed) {
      setConfirmOverwrite(true);
    } else {
      doInstall();
    }
  }

  if (loading) return <LoadingScreen onBack={onBack} title={slug} />;
  if (error || !detail) return <ErrorScreen onBack={onBack} message={error ?? "Not found"} />;

  const { skill, latestVersion, owner, moderation } = detail;
  const isSuspicious = moderation != null && (moderation.isSuspicious || moderation.isMalwareBlocked);
  const stats = skill.stats;
  const files = body ? _inferFiles(body) : [];

  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1 style={{ fontSize: 20 }}>{skill.displayName || slug}</h1>
      </div>

      <div className="detail-content">
        {/* Stats bar */}
        <div className="ch-stats-bar">
          {stats.downloads > 0 && (
            <span className="ch-stat">
              <Download size={13} /> {_fmt(stats.downloads)}
            </span>
          )}
          {stats.installsCurrent > 0 && (
            <span className="ch-stat">
              <Users size={13} /> {_fmt(stats.installsCurrent)}
            </span>
          )}
          {stats.stars > 0 && (
            <span className="ch-stat">
              <Star size={13} /> {stats.stars}
            </span>
          )}
          <span className={`ch-stat ch-stat--${isSuspicious ? "warn" : "ok"}`}>
            {isSuspicious ? <AlertTriangle size={13} /> : null}
            {isSuspicious ? "Suspicious" : "Clean"}
          </span>
        </div>

        {/* Meta */}
        <div className="settings-section">
          <div className="settings-card">
            <div className="settings-row">
              <span className="settings-row-label">Author</span>
              <span className="settings-row-value">{owner.displayName || owner.handle}</span>
            </div>
            <div className="settings-row">
              <span className="settings-row-label">Version</span>
              <span className="settings-row-value">v{latestVersion.version}</span>
            </div>
            {stats.versions > 1 && (
              <div className="settings-row">
                <span className="settings-row-label">Versions</span>
                <span className="settings-row-value">{stats.versions}</span>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        {skill.summary && (
          <div className="settings-section">
            <div className="settings-section-label">Description</div>
            <div className="settings-card">
              <div className="settings-row" style={{ alignItems: "flex-start", paddingTop: 14, paddingBottom: 14 }}>
                <span className="settings-row-label" style={{ lineHeight: 1.5 }}>{skill.summary}</span>
              </div>
            </div>
          </div>
        )}

        {/* Body */}
        {body && (
          <div className="settings-section">
            <div className="settings-section-label">Instructions</div>
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{_stripFrontmatter(body)}</ReactMarkdown>
            </div>
          </div>
        )}

        {/* Files declared in frontmatter */}
        {files.length > 0 && (
          <div className="settings-section">
            <div className="settings-section-label">Files</div>
            <div className="settings-card">
              {files.map((f) => (
                <button
                  key={f}
                  className="file-row"
                  onClick={() =>
                    onNavigate({
                      page: "file-viewer",
                      skillName: slug,
                      filePath: f,
                      source: "clawhub",
                      slug,
                    })
                  }
                >
                  <File size={15} color="var(--hint)" />
                  <span className="file-row-name">{f}</span>
                  <ChevronLeft size={14} color="var(--hint)" style={{ transform: "rotate(180deg)" }} />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Install button */}
        <div style={{ padding: "8px 0 40px" }}>
          <button
            className="btn-primary"
            style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
            onClick={handleInstallClick}
            disabled={installing}
          >
            {installing ? "Installing…" : installed ? (
              <><RefreshCw size={15} /> Update</>
            ) : (
              <><Download size={15} /> Install</>
            )}
          </button>
        </div>
      </div>

      {confirmOverwrite && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">Update "{slug}"?</h3>
            <p className="modal-body">
              This will overwrite your local copy. Any edits you made will be lost.
            </p>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setConfirmOverwrite(false)}>
                Cancel
              </button>
              <button className="btn-primary" onClick={doInstall} disabled={installing}>
                {installing ? "Updating…" : "Update"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared screens
// ---------------------------------------------------------------------------

function LoadingScreen({ onBack, title }: { onBack: () => void; title: string }) {
  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1 style={{ fontSize: 20 }}>{title}</h1>
      </div>
      <div className="centered">
        <div className="spinner" />
      </div>
    </div>
  );
}

function ErrorScreen({ onBack, message }: { onBack: () => void; message: string }) {
  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1 style={{ fontSize: 20 }}>Error</h1>
      </div>
      <div className="centered">
        <p style={{ color: "var(--destructive)", fontSize: 14 }}>{message}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _stripFrontmatter(raw: string): string {
  if (!raw.startsWith("---")) return raw;
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return raw;
  return raw.slice(end + 4).trim();
}

function _inferFiles(raw: string): string[] {
  if (!raw.startsWith("---")) return [];
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return [];
  const fm = raw.slice(3, end);
  const matches = fm.match(/files:\s*\[([^\]]*)\]/);
  if (!matches) return [];
  return matches[1]
    .split(",")
    .map((s) => s.trim().replace(/['"]/g, ""))
    .filter(Boolean);
}

function _fmt(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}
