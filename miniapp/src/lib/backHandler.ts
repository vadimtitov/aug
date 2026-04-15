import { createContext } from "react";

// A mutable ref that pages can override when they have unsaved edits.
// App.tsx wires the Telegram BackButton to call backHandlerRef.current().
// Pages swap in a safeBack function on mount and restore onBack on unmount.
export const BackHandlerContext = createContext<React.MutableRefObject<() => void>>(
  { current: () => {} }
);
