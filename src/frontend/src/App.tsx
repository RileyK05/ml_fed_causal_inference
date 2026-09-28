import { useState } from "react";
import Playground from "./components/Playground";
import Runs from "./components/Runs";
import Explain from "./components/Explain";
import { ThemeProvider, useTheme } from "./theme";

type Tab = "playground" | "explain" | "runs";

function Shell() {
  const [tab, setTab] = useState<Tab>("playground");
  const { theme, toggle } = useTheme();
  const nav = (id: Tab, label: string) => (
    <button className={tab === id ? "btn active" : "btn"} onClick={() => setTab(id)}>{label}</button>
  );
  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>fedci</h1>
          <p className="tagline">Q1 estimation playground · FOMC causal inference</p>
        </div>
        <nav>
          {nav("playground", "Playground")}
          {nav("explain", "Explain DML")}
          {nav("runs", "Runs")}
          <button className="btn" onClick={toggle} title="toggle light/dark">
            {theme === "light" ? "◐ dark" : "◑ light"}
          </button>
        </nav>
      </header>
      {tab === "playground" ? <Playground /> : tab === "explain" ? <Explain /> : <Runs />}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Shell />
    </ThemeProvider>
  );
}
