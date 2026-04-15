import { useEffect, useState } from "react";
import { ChevronLeft, Pencil, Check, X } from "lucide-react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import bash from "react-syntax-highlighter/dist/esm/languages/hljs/bash";
import javascript from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import markdown from "react-syntax-highlighter/dist/esm/languages/hljs/markdown";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { clawhubGetFile, deleteSkillFile, getSkillFile, updateSkillFile } from "../api.ts";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("markdown", markdown);

interface Props {
  skillName: string;
  filePath: string;
  source: "local" | "clawhub";
  slug?: string;
  onBack: () => void;
  onDeleted: () => void;
}

export function FileViewerPage({ skillName, filePath, source, slug, onBack, onDeleted }: Props) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [saving, setSaving] = useState(false);

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const fileName = filePath.split("/").pop() ?? filePath;
  const language = _detectLanguage(filePath);
  const readonly = source === "clawhub";

  useEffect(() => {
    const fetcher =
      source === "local"
        ? getSkillFile(skillName, filePath)
        : clawhubGetFile(slug!, filePath);

    fetcher
      .then((c) => {
        setContent(c);
        setEditContent(c);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [skillName, filePath, source, slug]);

  function startEdit() {
    setEditContent(content!);
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setEditContent(content!);
  }

  async function saveEdit() {
    setSaving(true);
    try {
      await updateSkillFile(skillName, filePath, editContent);
      setContent(editContent);
      setEditing(false);
    } catch (e) {
      alert(`Failed to save: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteSkillFile(skillName, filePath);
      onDeleted();
    } catch (e) {
      alert(`Failed to delete: ${e}`);
      setDeleting(false);
      setConfirmDelete(false);
    }
  }

  return (
    <div className="screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} />
          Back
        </button>
        <h1 style={{ fontSize: 18, fontFamily: "monospace" }}>{fileName}</h1>
        {!readonly && !editing && content !== null && (
          <button className="icon-action-btn" onClick={startEdit}>
            <Pencil size={18} color="var(--brand)" />
          </button>
        )}
        {!readonly && editing && (
          <div style={{ display: "flex", gap: 8 }}>
            <button className="icon-action-btn" onClick={cancelEdit} disabled={saving}>
              <X size={18} color="var(--hint)" />
            </button>
            <button className="icon-action-btn" onClick={saveEdit} disabled={saving}>
              <Check size={18} color="var(--brand)" />
            </button>
          </div>
        )}
      </div>

      {loading && (
        <div className="centered">
          <div className="spinner" />
        </div>
      )}

      {error && (
        <div className="centered">
          <p style={{ color: "var(--destructive)", fontSize: 14 }}>{error}</p>
        </div>
      )}

      {!loading && !error && content !== null && (
        <>
          {editing ? (
            <textarea
              className="edit-textarea edit-textarea--full"
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
            />
          ) : (
            <div className="file-content">
              <SyntaxHighlighter
                language={language}
                style={atomOneDark}
                customStyle={{
                  background: "transparent",
                  padding: "16px",
                  margin: 0,
                  fontSize: 13,
                  lineHeight: 1.6,
                }}
                wrapLongLines
              >
                {content}
              </SyntaxHighlighter>
            </div>
          )}

          {!readonly && !editing && (
            <div style={{ padding: "16px 16px 40px" }}>
              <button className="btn-danger" onClick={() => setConfirmDelete(true)}>
                Delete File
              </button>
            </div>
          )}
        </>
      )}

      {confirmDelete && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">Delete "{fileName}"?</h3>
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

function _detectLanguage(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    py: "python",
    sh: "bash",
    bash: "bash",
    js: "javascript",
    ts: "javascript",
    md: "markdown",
  };
  return map[ext] ?? "plaintext";
}
