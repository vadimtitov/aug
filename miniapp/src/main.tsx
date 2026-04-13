import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { mockTelegramEnv, isTMA } from "@telegram-apps/sdk-react";
import App from "./App.tsx";
import "./index.css";

// In local dev (non-Telegram context) inject a mock environment.
// auth_date uses the current timestamp so it doesn't expire during the session.
// The backend accepts this in DEBUG=True mode (HMAC verification is skipped).
if (!isTMA()) {
  const devInitData = new URLSearchParams({
    auth_date: String(Math.floor(Date.now() / 1000)),
    user: JSON.stringify({ id: 1, first_name: "Dev", username: "devuser" }),
    signature: "dev_bypass",
    hash: "dev_bypass",
  }).toString();

  mockTelegramEnv({
    launchParams: new URLSearchParams([
      ["tgWebAppVersion", "8"],
      ["tgWebAppPlatform", "tdesktop"],
      ["tgWebAppData", devInitData],
      [
        "tgWebAppThemeParams",
        JSON.stringify({
          bg_color: "#000000",
          text_color: "#f5f5f5",
          hint_color: "#8e8e93",
          link_color: "#0a84ff",
          button_color: "#0a84ff",
          button_text_color: "#ffffff",
          secondary_bg_color: "#1c1c1e",
          header_bg_color: "#000000",
          accent_text_color: "#0a84ff",
          section_bg_color: "#1c1c1e",
          section_header_text_color: "#8e8e93",
          subtitle_text_color: "#8e8e93",
          destructive_text_color: "#ff453a",
        }),
      ],
    ]),
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
