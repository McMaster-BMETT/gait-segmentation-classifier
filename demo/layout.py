"""Dash layout — page structure and all HTML/dcc components."""

from dash import html, dcc

from .constants import STAGE_DISPLAY, STAGE_ORDER, METRIC_DEFS, METRIC_ORDER, PHASE_COLORS


def build_layout(demo_data):
    """Build the complete single-page layout.

    Args:
        demo_data: loaded demo_data.json dict

    Returns:
        dash html.Div
    """
    model_info = demo_data.get("model_info", {})

    return html.Div(className="dashboard-container", children=[

        # ── Section 1: Hero ──────────────────────────────
        html.Div(className="hero-section", children=[
            html.H1("Gait Phase Segmentation", className="hero-title"),
            html.P(
                "Real-time knee rehabilitation assessment using wearable sensors",
                className="hero-subtitle",
            ),
            html.Div(className="hero-badges", children=[
                html.Span("8-channel sensor", className="hero-badge"),
                html.Span("50 Hz", className="hero-badge"),
                html.Span("6 gait phases", className="hero-badge"),
                html.Span("BiTCN classifier", className="hero-badge"),
            ]),
        ]),

        # ── Section 2: Signal + Segmentation Viewer ──────
        html.Div(className="section", children=[
            html.H2("Signal Viewer", className="section-title"),

            # Stage selector pills
            html.Div(id="stage-selector", className="stage-selector", children=[
                html.Button(
                    STAGE_DISPLAY[s],
                    id={"type": "stage-btn", "index": s},
                    className="stage-btn active" if i == 0 else "stage-btn",
                    n_clicks=0,
                )
                for i, s in enumerate(STAGE_ORDER)
            ]),

            # Hidden store for selected stage
            dcc.Store(id="selected-stage", data=STAGE_ORDER[0]),

            # Signal chart
            html.Div(className="chart-card", children=[
                dcc.Graph(id="signal-chart", config={"displayModeBar": False}),
                # Phase legend
                html.Div(className="phase-legend", children=[
                    html.Span(children=[
                        html.Span(className="phase-swatch",
                                  style={"background": color}),
                        html.Span(f"{abbrev} — {name}",
                                  style={"color": "#6B7280", "fontSize": "12px"}),
                    ], className="phase-legend-item")
                    for abbrev, name, color in [
                        ("IC", "Initial Contact", PHASE_COLORS[0]),
                        ("LR", "Loading Response", PHASE_COLORS[1]),
                        ("MSt", "Mid Stance", PHASE_COLORS[2]),
                        ("TSt", "Terminal Stance", PHASE_COLORS[3]),
                        ("ISw", "Initial Swing", PHASE_COLORS[4]),
                        ("TSw", "Terminal Swing", PHASE_COLORS[5]),
                    ]
                ]),
            ]),
        ]),

        # ── Section 3: Clinical Metrics Cards ────────────
        html.Div(className="section", children=[
            html.H2("Clinical Metrics", className="section-title"),

            html.Div(id="metrics-grid", className="metrics-grid", children=[
                _metric_card_placeholder(key) for key in METRIC_ORDER
            ]),

            # Radar chart
            html.Div(className="chart-card", children=[
                dcc.Graph(id="radar-chart", config={"displayModeBar": False}),
            ]),
        ]),

        # ── Section 4: Recovery Progression (one chart per metric) ──
        html.Div(className="section", children=[
            html.H2("Recovery Progression", className="section-title"),
            html.Div(className="progression-grid", children=[
                html.Div(className="chart-card", children=[
                    dcc.Graph(
                        id=f"progression-{key}",
                        config={"displayModeBar": False},
                    ),
                ])
                for key in METRIC_ORDER
            ]),
        ]),

        # ── Section 5: Model Info Footer ─────────────────
        html.Div(className="model-footer", children=[
            html.H2("Model Architecture", className="section-title"),
            html.P(
                f"Input(8ch) → Projection({model_info.get('hidden_channels', 64)}) "
                f"→ {model_info.get('num_blocks', 4)} TCN Blocks "
                f"(dilations {model_info.get('dilations', [1,2,4,8])}) "
                f"→ Output(6 phases)",
                className="model-arch",
            ),
            html.Div(className="stats-grid", children=[
                _stat_item(f"{model_info.get('total_params', 0):,}", "Parameters"),
                _stat_item(
                    f"{model_info.get('accuracy', 0) * 100:.1f}%"
                    if model_info.get("accuracy") else "—",
                    "Accuracy",
                ),
                _stat_item(
                    str(sum(2 * d * (model_info.get("kernel_size", 3) - 1)
                            for d in model_info.get("dilations", [1, 2, 4, 8])) + 1),
                    "Receptive Field",
                ),
                _stat_item("< 5 ms", "Inference Time"),
            ]),
        ]),
    ])


def _metric_card_placeholder(key):
    """Create a metric card with placeholder values (filled by callback)."""
    mdef = METRIC_DEFS[key]
    return html.Div(className="metric-card", id={"type": "metric-card", "index": key}, children=[
        html.Div(className="metric-dot", id={"type": "metric-dot", "index": key}),
        html.Div(mdef["short"], className="metric-label"),
        html.Div("—", className="metric-value", id={"type": "metric-value", "index": key}),
        html.Div(f"Target: {mdef['target']}", className="metric-target"),
    ])


def _stat_item(value, label):
    return html.Div(className="stat-item", children=[
        html.Div(value, className="stat-value"),
        html.Div(label, className="stat-label"),
    ])
