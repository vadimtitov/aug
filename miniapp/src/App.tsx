import { useEffect, useState } from "react";
import { retrieveRawInitData } from "@telegram-apps/sdk-react";
import { initAuth } from "./api.ts";
import { HomePage } from "./pages/HomePage.tsx";
import { SettingsPage } from "./pages/SettingsPage.tsx";

type AuthState = "pending" | "ready" | "error";
type Page = "home" | "settings";

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("pending");
  const [page, setPage] = useState<Page>("home");

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

  if (page === "settings") {
    return <SettingsPage onBack={() => setPage("home")} />;
  }

  return (
    <HomePage
      onNavigate={(dest) => setPage(dest)}
    />
  );
}
