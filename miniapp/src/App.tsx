import { useEffect, useState } from "react";
import { retrieveRawInitData } from "@telegram-apps/sdk-react";
import { initAuth } from "./api.ts";
import { HomePage } from "./pages/HomePage.tsx";
import { SettingsPage } from "./pages/SettingsPage.tsx";
import { SkillsPage } from "./pages/SkillsPage.tsx";
import { SkillDetailPage } from "./pages/SkillDetailPage.tsx";
import { FileViewerPage } from "./pages/FileViewerPage.tsx";
import type { PageState } from "./types.ts";

type AuthState = "pending" | "ready" | "error";

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("pending");
  const [stack, setStack] = useState<PageState[]>([{ page: "home" }]);

  const current = stack[stack.length - 1];

  function navigate(state: PageState) {
    setStack((s) => [...s, state]);
  }

  function goBack() {
    setStack((s) => (s.length > 1 ? s.slice(0, -1) : s));
  }

  useEffect(() => {
    async function authenticate() {
      try {
        const initDataRaw = retrieveRawInitData();
        if (!initDataRaw) throw new Error("No initData available");
        await initAuth(initDataRaw);
        setAuthState("ready");
      } catch (e) {
        console.error("Auth failed", e);
        setAuthState("error");
      }
    }
    authenticate();
  }, []);

  if (authState === "pending") {
    return (
      <div className="centered">
        <div className="spinner" />
        <p style={{ color: "var(--hint)", fontSize: 15 }}>Authenticating…</p>
      </div>
    );
  }

  if (authState === "error") {
    return (
      <div className="centered">
        <h2>Authentication failed</h2>
        <p>Open this app from your Telegram bot.</p>
      </div>
    );
  }

  if (current.page === "settings") {
    return <SettingsPage onBack={goBack} />;
  }

  if (current.page === "skills") {
    return <SkillsPage onBack={goBack} onNavigate={navigate} />;
  }

  if (current.page === "skill-detail") {
    return (
      <SkillDetailPage
        skillName={current.skillName}
        source={current.source}
        slug={current.source === "clawhub" ? current.slug : undefined}
        onBack={goBack}
        onNavigate={navigate}
        onDeleted={goBack}
      />
    );
  }

  if (current.page === "file-viewer") {
    return (
      <FileViewerPage
        skillName={current.skillName}
        filePath={current.filePath}
        source={current.source}
        slug={current.source === "clawhub" ? current.slug : undefined}
        onBack={goBack}
        onDeleted={goBack}
      />
    );
  }

  return (
    <HomePage
      onNavigate={(dest) => {
        if (dest === "settings") navigate({ page: "settings" });
        if (dest === "skills") navigate({ page: "skills" });
      }}
    />
  );
}
