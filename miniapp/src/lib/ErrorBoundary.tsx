import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches render-time exceptions so a single bad value can't blank the whole
 * mini app. Without this, an uncaught throw unmounts the entire React tree and
 * the WebView shows a black screen with no recourse but to close and reopen.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Render error", error, info.componentStack);
  }

  handleReset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="centered">
        <h2>Something went wrong</h2>
        <p style={{ color: "var(--hint)", fontSize: 14 }}>{this.state.error.message}</p>
        <button className="btn-secondary" onClick={this.handleReset}>
          Try again
        </button>
      </div>
    );
  }
}
