import type { ReactNode } from "react";

type NavItem = {
  id: string;
  label: string;
};

type AppShellProps = {
  navItems: NavItem[];
  activeView: string;
  onNavigate: (view: string) => void;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  onNewBatch: () => void;
  healthState: "checking" | "healthy" | "offline";
  eyebrow: string;
  title: string;
  description: string;
  searchPanel?: ReactNode;
  children: ReactNode;
};

export function AppShell({
  navItems,
  activeView,
  onNavigate,
  searchQuery,
  onSearchChange,
  onNewBatch,
  healthState,
  eyebrow,
  title,
  description,
  searchPanel,
  children,
}: AppShellProps) {
  return (
    <main className="dashboard-shell">
      <div className="dashboard-noise" aria-hidden="true" />

      <aside className="sidebar">
        <div className="sidebar-brand">
          <p className="sidebar-kicker">PData</p>
          <strong className="sidebar-title">Control Room</strong>
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={item.id === activeView ? "sidebar-link sidebar-link-active" : "sidebar-link"}
              onClick={() => onNavigate(item.id)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <section className="dashboard-main">
        <header className="topbar">
          <label className="topbar-search" aria-label="Search">
            <span className="topbar-search-label">Quick Search</span>
            <input
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="Search datasets, runs, documents"
              type="text"
              value={searchQuery}
            />
          </label>

          <div className="topbar-actions">
            <button className="topbar-button" onClick={onNewBatch} type="button">
              <span className="topbar-button-mark" aria-hidden="true">
                +
              </span>
              <span>New Batch</span>
            </button>
          </div>
        </header>

        {searchPanel}

        <header className="dashboard-header">
          <div>
            <p className="dashboard-kicker">{eyebrow}</p>
            <h1 className="dashboard-title">{title}</h1>
            <p className="dashboard-copy">{description}</p>
          </div>

          <div
            className={
              healthState === "healthy"
                ? "dashboard-badge dashboard-badge-healthy"
                : healthState === "offline"
                  ? "dashboard-badge dashboard-badge-offline"
                  : "dashboard-badge dashboard-badge-checking"
            }
          >
            <span className="status-dot" aria-hidden="true" />
            {healthState === "healthy"
              ? "API Healthy"
              : healthState === "offline"
                ? "API Offline"
                : "Checking API"}
          </div>
        </header>

        {children}
      </section>
    </main>
  );
}
