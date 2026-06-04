/**
 * Live browser view — watch the agent drive Chrome in real time.
 *
 * Opens a WebSocket to /browser/stream and paints each JPEG frame to a canvas.
 * Frames are dropped-to-latest on the client too: while one frame is decoding,
 * any newer frames replace the queued one, so we always render the freshest
 * frame and never build up lag on a slow link.
 *
 * The screencast is event-driven on the server: an idle page sends nothing, so
 * gaps between frames are normal while the agent is thinking. The connection
 * auto-reconnects with backoff if it drops.
 */

import { useEffect, useRef, useState } from "react";
import { ChevronLeft } from "lucide-react";
import { browserStreamProtocols, browserStreamUrl, getBrowserStatus } from "../api.ts";

type Status = "connecting" | "live" | "reconnecting" | "unavailable" | "error";

// Application WebSocket close codes set by the backend (see routers/browser.py).
const CLOSE_UNAUTHORIZED = 4401;
const CLOSE_UNAVAILABLE = 4404;

const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 15000;

interface BrowserPageProps {
  onBack: () => void;
}

export function BrowserPage({ onBack }: BrowserPageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<Status>("connecting");

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let backoff = RECONNECT_MIN_MS;
    let closedByUs = false;

    // Drop-to-latest decode pipeline: hold at most one pending frame.
    let pending: Blob | null = null;
    let decoding = false;

    function paint(bitmap: ImageBitmap) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      const ctx = canvas.getContext("2d");
      ctx?.drawImage(bitmap, 0, 0);
      bitmap.close();
    }

    function pump() {
      if (decoding || !pending) return;
      decoding = true;
      const blob = pending;
      pending = null;
      createImageBitmap(blob)
        .then((bitmap) => {
          paint(bitmap);
          decoding = false;
          pump();
        })
        .catch(() => {
          decoding = false;
          pump();
        });
    }

    function scheduleReconnect() {
      if (closedByUs) return;
      setStatus("reconnecting");
      reconnectTimer = setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, RECONNECT_MAX_MS);
    }

    function connect() {
      try {
        ws = new WebSocket(browserStreamUrl(), browserStreamProtocols());
      } catch {
        setStatus("error");
        return;
      }
      ws.binaryType = "blob";
      ws.onopen = () => setStatus("connecting"); // still waiting for first frame
      ws.onmessage = (event) => {
        backoff = RECONNECT_MIN_MS; // healthy traffic resets backoff
        setStatus("live");
        pending = event.data as Blob;
        pump();
      };
      ws.onclose = (event) => {
        if (event.code === CLOSE_UNAUTHORIZED) {
          closedByUs = true;
          setStatus("error");
          return;
        }
        if (event.code === CLOSE_UNAVAILABLE) {
          closedByUs = true;
          setStatus("unavailable");
          return;
        }
        scheduleReconnect();
      };
      ws.onerror = () => ws?.close();
    }

    // Probe availability first so we can show a clear message instead of a
    // silent reconnect loop when the browser tool isn't configured at all.
    getBrowserStatus()
      .then(({ available }) => {
        if (closedByUs) return;
        if (!available) {
          setStatus("unavailable");
          return;
        }
        connect();
      })
      .catch(() => {
        if (!closedByUs) connect();
      });

    return () => {
      closedByUs = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return (
    <div className="screen browser-screen">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={20} strokeWidth={2.5} />
          Back
        </button>
        <h1>Browser</h1>
      </div>

      <div className="browser-stage">
        <canvas ref={canvasRef} className="browser-canvas" />
        {status !== "live" && (
          <div className="browser-overlay">
            {(status === "connecting" || status === "reconnecting") && (
              <div className="spinner" />
            )}
            <p>{statusMessage(status)}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function statusMessage(status: Status): string {
  switch (status) {
    case "connecting":
      return "Connecting to the browser…";
    case "reconnecting":
      return "Reconnecting…";
    case "unavailable":
      return "The browser isn't available right now.";
    case "error":
      return "Couldn't open the browser view.";
    default:
      return "";
  }
}
