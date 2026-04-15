import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { retrieveRawInitData } from "@telegram-apps/sdk-react";
import { initAuth } from "./api.ts";
import { HomePage } from "./pages/HomePage.tsx";
import { SettingsPage } from "./pages/SettingsPage.tsx";
import { SkillsPage } from "./pages/SkillsPage.tsx";
import type { PageState } from "./types.ts";
import { isInTelegram, tg } from "./lib/tg.ts";
import { BackHandlerContext } from "./lib/backHandler.ts";

const SkillDetailPage = lazy(() => import("./pages/SkillDetailPage.tsx").then((m) => ({ default: m.SkillDetailPage })));
const FileViewerPage = lazy(() => import("./pages/FileViewerPage.tsx").then((m) => ({ default: m.FileViewerPage })));

type AuthState = "pending" | "ready" | "error";

const LazyFallback = () => (
  <div className="centered">
    <div className="spinner" />
  </div>
);

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("pending");
  const [stack, setStack] = useState<PageState[]>([{ page: "home" }]);

  const current = stack[stack.length - 1];

  // Mutable ref that deep pages can override when they have unsaved edits.
  // The Telegram BackButton calls backHandlerRef.current() instead of goBack directly.
  const backHandlerRef = useRef<() => void>(() => {});

  function navigate(state: PageState) {
    setStack((s) => [...s, state]);
  }

  function goBack() {
    setStack((s) => (s.length > 1 ? s.slice(0, -1) : s));
  }

  // Keep backHandlerRef pointing at goBack unless a page overrides it
  useEffect(() => {
    backHandlerRef.current = goBack;
  });

  // Telegram BackButton — show when navigated away from home
  useEffect(() => {
    const btn = tg()?.BackButton;
    if (!btn) return;
    const handler = () => backHandlerRef.current();
    if (stack.length > 1) {
      btn.show();
      btn.onClick(handler);
    } else {
      btn.hide();
    }
    return () => btn.offClick(handler);
  }, [stack.length]);

  useEffect(() => {
    tg()?.disableVerticalSwipes?.();
    if (isInTelegram()) document.body.classList.add("in-telegram");
  }, []);

  // Left-edge swipe to go back
  useEffect(() => {
    let startX = 0;
    function onTouchStart(e: TouchEvent) {
      startX = e.touches[0].clientX;
    }
    function onTouchEnd(e: TouchEvent) {
      const dx = e.changedTouches[0].clientX - startX;
      if (startX < 20 && dx > 60) backHandlerRef.current();
    }
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    return () => {
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

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

  if (current.page === "home") {
    return (
      <HomePage
        onNavigate={(dest) => {
          if (dest === "settings") navigate({ page: "settings" });
          if (dest === "skills") navigate({ page: "skills" });
        }}
      />
    );
  }

  if (current.page === "settings") {
    return <SettingsPage onBack={goBack} />;
  }

  if (current.page === "skills") {
    return <SkillsPage onBack={goBack} onNavigate={navigate} />;
  }

  return (
    <BackHandlerContext.Provider value={backHandlerRef}>
      <Suspense fallback={<LazyFallback />}>
        {current.page === "skill-detail" && (
          <SkillDetailPage
            skillName={current.skillName}
            source={current.source}
            slug={current.source === "clawhub" ? current.slug : undefined}
            onBack={goBack}
            onNavigate={navigate}
            onDeleted={goBack}
          />
        )}
        {current.page === "file-viewer" && (
          <FileViewerPage
            skillName={current.skillName}
            filePath={current.filePath}
            source={current.source}
            slug={current.source === "clawhub" ? current.slug : undefined}
            onBack={goBack}
            onDeleted={goBack}
          />
        )}
      </Suspense>
    </BackHandlerContext.Provider>
  );
}
