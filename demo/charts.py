"""Plotly figure builders for the demo dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import (
    PHASE_COLORS, STAGE_DISPLAY, STAGE_ORDER, METRIC_DEFS, METRIC_ORDER,
    METRIC_LINE_COLORS, EMG_QUAD_COLOR, EMG_HAM_COLOR, KNEE_FLEXION_COLOR,
    BG_COLOR, GRID_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, CARD_BG,
)

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY),
    margin=dict(l=48, r=16, t=24, b=36),
)


def _style_axis(fig, row_count):
    """Apply consistent axis styling across subplots."""
    for i in range(1, row_count + 1):
        yaxis = f"yaxis{i}" if i > 1 else "yaxis"
        xaxis = f"xaxis{i}" if i > 1 else "xaxis"
        fig.update_layout(**{
            yaxis: dict(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            xaxis: dict(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11, color=TEXT_SECONDARY)),
        })


def build_signal_figure(stage_data, phase_names):
    """Build the 3-subplot signal + segmentation figure.

    Args:
        stage_data: dict with knee_flexion, emg_quad, emg_ham, true_labels, pred_labels
        phase_names: list of 6 phase display names

    Returns:
        plotly.graph_objects.Figure
    """
    knee = stage_data["knee_flexion"]
    eq = stage_data["emg_quad"]
    eh = stage_data["emg_ham"]
    pred = stage_data["pred_labels"]
    true = stage_data["true_labels"]
    n = len(knee)
    t = [i / 50.0 for i in range(n)]  # time axis in seconds

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=["Knee Flexion Angle", "EMG Signals", "True vs Predicted"],
    )

    # ── Row 1: knee flexion with phase-colored background bands ──
    fig.add_trace(
        go.Scatter(x=t, y=knee, mode="lines",
                   line=dict(color=KNEE_FLEXION_COLOR, width=1.5),
                   name="Knee Flexion", showlegend=False),
        row=1, col=1,
    )
    # Add phase background bands
    _add_phase_bands(fig, t, pred, phase_names, row=1)

    # ── Row 2: EMG quad + hamstring ──
    fig.add_trace(
        go.Scatter(x=t, y=eq, mode="lines",
                   line=dict(color=EMG_QUAD_COLOR, width=1.2),
                   name="Quadriceps"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=eh, mode="lines",
                   line=dict(color=EMG_HAM_COLOR, width=1.2),
                   name="Hamstring"),
        row=2, col=1,
    )

    # ── Row 3: True vs Predicted label strips ──
    _add_label_strip(fig, t, true, y_offset=1, label="True", phase_names=phase_names, row=3)
    _add_label_strip(fig, t, pred, y_offset=0, label="Pred", phase_names=phase_names, row=3)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Angle (°)", row=1, col=1)
    fig.update_yaxes(title_text="ADC", row=2, col=1)
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["Pred", "True"],
        range=[-0.5, 1.5], row=3, col=1,
    )
    _style_axis(fig, 3)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=13, color=TEXT_SECONDARY, family="Inter, sans-serif"))

    return fig


def _add_phase_bands(fig, t, labels, phase_names, row):
    """Add colored vertical bands for each predicted gait phase."""
    n = len(labels)
    i = 0
    added_phases = set()
    while i < n:
        phase = int(labels[i])
        j = i
        while j < n and int(labels[j]) == phase:
            j += 1
        show = phase not in added_phases
        added_phases.add(phase)
        fig.add_vrect(
            x0=t[i], x1=t[min(j, n) - 1],
            fillcolor=PHASE_COLORS[phase % len(PHASE_COLORS)],
            opacity=0.12, line_width=0,
            row=row, col=1,
        )
        if show and (j - i) > 3:
            mid = (t[i] + t[min(j, n) - 1]) / 2
            fig.add_annotation(
                x=mid, y=1.0, yref=f"y{row} domain" if row > 1 else "y domain",
                text=phase_names[phase] if phase < len(phase_names) else "",
                showarrow=False, font=dict(size=9, color=PHASE_COLORS[phase % len(PHASE_COLORS)]),
                yshift=10, row=row, col=1,
            )
        i = j


def _add_label_strip(fig, t, labels, y_offset, label, phase_names, row):
    """Add a colored block strip for phase labels."""
    n = len(labels)
    i = 0
    while i < n:
        phase = int(labels[i])
        j = i
        while j < n and int(labels[j]) == phase:
            j += 1
        color = PHASE_COLORS[phase % len(PHASE_COLORS)]
        fig.add_trace(
            go.Bar(
                x=[t[min(j, n) - 1] - t[i]],
                y=[1],
                base=[y_offset - 0.4],
                marker_color=color,
                orientation="v",
                width=t[min(j, n) - 1] - t[i],
                showlegend=False,
                hovertext=f"{label}: {phase_names[phase] if phase < len(phase_names) else phase}",
                hoverinfo="text",
            ),
            row=row, col=1,
        )
        # Shift bar to correct x position
        fig.data[-1].x = [(t[i] + t[min(j, n) - 1]) / 2]
        i = j


def build_radar_chart(metrics, stage_name):
    """Build a radar chart of clinical metrics normalized to targets.

    Args:
        metrics: dict with qii, hq_functional, swing_peak_flexion, lsi_temporal
        stage_name: stage key for title

    Returns:
        plotly.graph_objects.Figure
    """
    categories = []
    values = []

    for key in METRIC_ORDER:
        mdef = METRIC_DEFS[key]
        val = metrics.get(key)
        if val is None:
            val = 0
        # Normalize to 0-1 scale
        if mdef.get("radar_invert"):
            # Lower is better: 0.0 → 1.0, radar_max → 0.0
            norm = max(0, 1 - val / mdef["radar_max"])
        else:
            norm = min(1, val / mdef["radar_max"])
        categories.append(mdef["short"])
        values.append(round(norm, 3))

    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=5, color="#2563EB"),
        name=STAGE_DISPLAY.get(stage_name, stage_name),
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=360,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=GRID_COLOR, tickfont=dict(size=10, color=TEXT_SECONDARY),
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=TEXT_PRIMARY),
                gridcolor=GRID_COLOR,
            ),
        ),
        showlegend=False,
    )
    return fig


def build_progression_chart(all_stages_data):
    """Build a line chart showing metric progression across healing stages.

    Args:
        all_stages_data: dict[stage_name] → dict with "metrics" key

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    x_labels = [STAGE_DISPLAY.get(s, s) for s in STAGE_ORDER]

    for key in METRIC_ORDER:
        mdef = METRIC_DEFS[key]
        y_vals = []
        for stage in STAGE_ORDER:
            sdata = all_stages_data.get(stage)
            val = sdata["metrics"].get(key) if sdata else None
            y_vals.append(val)

        fig.add_trace(go.Scatter(
            x=x_labels,
            y=y_vals,
            mode="lines+markers",
            name=mdef["short"],
            line=dict(color=METRIC_LINE_COLORS[key], width=2.5),
            marker=dict(size=7),
            connectgaps=True,
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=380,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=13, color=TEXT_PRIMARY)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11, color=TEXT_SECONDARY)),
    )
    fig.update_xaxes(title_text="Recovery Stage")
    fig.update_yaxes(title_text="Metric Value")

    return fig
