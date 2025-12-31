import React from "react";

export type AppShellProps = {
  /**
   * Current role (demo-only).
   * Today we still switch apps in `App.tsx` via local state.
   */
  role?: string | null;
  children: React.ReactNode;
};

/**
 * Minimal shared shell.
 *
 * Phase-1 goal: provide a stable "top-level layout system" landing zone
 * without changing any existing app UI/logic.
 *
 * Later phases can move shared header/sidebar/error-boundary into this shell,
 * and use per-app configs like `apps/<role>/navigation.ts` and
 * `apps/<role>/permissions.ts`.
 */
export function AppShell({ children }: AppShellProps) {
  return <div className="min-h-screen">{children}</div>;
}


