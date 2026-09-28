import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { useTheme } from "../theme";
import type { PlotFigure } from "../types";

export default function Plot({ fig }: { fig: PlotFigure }) {
  const ref = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const layout = {
      ...fig.layout,
      template: theme === "dark" ? "plotly_dark" : "plotly_white",
      paper_bgcolor: "rgba(0,0,0,0)",  // inherit the card background in both themes
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.react(el, fig.data, layout, { responsive: true, displaylogo: false });
    return () => Plotly.purge(el);  // capture the element: ref.current is nulled before this runs
  }, [fig, theme]);
  return <div ref={ref} className="plot" />;
}
