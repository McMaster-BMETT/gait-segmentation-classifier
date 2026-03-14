"""Gait Phase Segmentation — Expo Demo Dashboard.

Usage:
    1. Generate demo data:  python scripts/export_demo_data.py
    2. Launch dashboard:    python demo/app.py
    3. Open browser:        http://localhost:8050
"""

import json
import os
import sys

from dash import Dash, Input, Output

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.layout import build_layout
from demo.charts import build_progression_chart
from demo.callbacks import register_callbacks

DATA_PATH = os.path.join(os.path.dirname(__file__), "demo_data.json")


def create_app():
    app = Dash(
        __name__,
        title="Gait Phase Segmentation",
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    # Load pre-computed data
    with open(DATA_PATH) as f:
        demo_data = json.load(f)

    app.layout = build_layout(demo_data)

    # Register interactive callbacks
    register_callbacks(app, demo_data)

    # Pre-build the progression chart (static, not stage-dependent)
    @app.callback(
        Output("progression-chart", "figure"),
        Input("selected-stage", "data"),
    )
    def update_progression(_):
        return build_progression_chart(demo_data["stages"])

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8050, debug=False)
