"""Gradio-based demo application for FoodSeg103 semantic segmentation.

This module builds and launches a Gradio Blocks interface that allows users to:

<<<<<<< HEAD
* Select between different segmentation models (BiSeNetV1, BiSeNetV1 v4, CCNet).
=======
* Select between different segmentation models (BiSeNetV1, CCNet).
>>>>>>> 4b2263ec99bfd68c12e9df2ee5b7100f626d0ed2
* Upload images or videos for inference.
* Visualise overlaid segmentation masks, per-class pixel distributions,
  confidence metrics, and timing breakdowns.

The UI follows a dark-themed, premium design with custom CSS and
matplotlib charts rendered in matching colours.

Usage::

    python -m app.main [--host HOST] [--port PORT] [--share]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _ensure_numpy_compat() -> None:
    """Patch missing NumPy aliases for compatibility with older libraries.

    Some downstream packages (e.g. certain Gradio or timm releases)
    reference deprecated NumPy symbols like ``np.bool8``, ``np.float_``,
    ``np.complex_``, and ``np.obj2sctype``.  This shim re-adds them so
    the import chain does not break under NumPy >= 1.24.
    """
    import numpy as np

    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    if not hasattr(np, "obj2sctype"):
        def _obj2sctype(rep, default=None):
            try:
                return np.dtype(rep).type
            except Exception:
                return default

        np.obj2sctype = _obj2sctype


def _ensure_huggingface_hub_compat() -> None:
    """Patch a missing ``HfFolder`` class in newer ``huggingface_hub`` versions.

    Some model-loading utilities still rely on ``huggingface_hub.HfFolder``
    which was removed in recent releases.  This shim provides a minimal
    drop-in replacement backed by the current token API.
    """
    try:
        import huggingface_hub as hf_hub
    except ImportError:
        return

    if hasattr(hf_hub, "HfFolder"):
        return

    from huggingface_hub import constants

    token_path = Path(constants.HF_TOKEN_PATH)

    class HfFolder:
        @staticmethod
        def path_token() -> str:
            return str(token_path)

        @staticmethod
        def get_token():
            get_token = getattr(hf_hub, "get_token", None)
            return get_token() if callable(get_token) else None

        @staticmethod
        def save_token(token: str) -> None:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token, encoding="utf-8")

        @staticmethod
        def delete_token() -> None:
            if token_path.exists():
                token_path.unlink()

    hf_hub.HfFolder = HfFolder


def _ensure_gradio_schema_compat() -> None:
    """Patch Gradio client schema parser for boolean JSON-schema branches.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """
    try:
        from gradio_client import utils as gr_client_utils
    except ImportError:
        return

    target_name = "_json_schema_to_python_type"
    original = getattr(gr_client_utils, target_name, None)
    if original is None:
        return

    if getattr(original, "__foodseg_bool_schema_patch__", False):
        return

    def _patched_json_schema_to_python_type(schema, defs=None):
        # Gradio <-> pydantic schema mismatch: schema can be plain bool.
        if isinstance(schema, bool):
            return "Any" if schema else "None"

        if isinstance(schema, dict):
            additional = schema.get("additionalProperties")
            if isinstance(additional, bool):
                # Normalize to a dict-like branch expected by older parser code.
                schema = dict(schema)
                schema["additionalProperties"] = {} if additional else {"type": "null"}

        return original(schema, defs)

    _patched_json_schema_to_python_type.__foodseg_bool_schema_patch__ = True
    setattr(gr_client_utils, target_name, _patched_json_schema_to_python_type)


_ensure_numpy_compat()
_ensure_huggingface_hub_compat()
_ensure_gradio_schema_compat()

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from app.service import FoodSegDemoService


# ─── Chart Helpers ──────────────────────────────────────────────────

_BG        = "#0f172a"
_CARD_BG   = "#1e293b"
_TEXT      = "#e2e8f0"
_SUBTEXT   = "#94a3b8"
_ACCENT    = "#10b981"
_ACCENT2   = "#6ee7b7"
_WARN      = "#f59e0b"
_BAR_COLORS = ["#10b981", "#06b6d4", "#8b5cf6", "#f59e0b", "#ef4444", "#ec4899", "#14b8a6", "#f97316"]


def _apply_chart_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply the dark theme to a matplotlib axes/figure."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD_BG)
    ax.tick_params(colors=_SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _build_metrics_chart(metrics: dict) -> plt.Figure:
    """Build a multi-panel metrics figure from the prediction metrics dict."""
    prediction = metrics.get("prediction", {})
    seg_metrics = metrics.get("segmentation_metrics", {})
    has_seg = seg_metrics.get("available", True) and "aAcc" in seg_metrics

    ncols = 3 if has_seg else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 3.8))
    if ncols == 1:
        axes = [axes]

    # ── Panel 1: Top predicted classes (horizontal bar) ──
    ax = axes[0]
    top_classes = prediction.get("top_classes", [])
    if top_classes:
        names = [c["class_name"][:18] for c in reversed(top_classes[:8])]
        ratios = [c["ratio"] * 100 for c in reversed(top_classes[:8])]
        colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(names))]
        bars = ax.barh(names, ratios, color=colors, height=0.6, edgecolor="none")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Pixel %")
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", color=_SUBTEXT, fontsize=7)
    ax.set_title("Top Predicted Classes", fontsize=10, fontweight="bold", pad=10)
    _apply_chart_style(ax, fig)

    # ── Panel 2: Confidence summary ──
    ax2 = axes[1]
    conf_labels = ["Mean", "Max", "Min"]
    conf_values = [
        prediction.get("mean_confidence", 0),
        prediction.get("max_confidence", 0),
        prediction.get("min_confidence", 0),
    ]
    bar_colors = [_ACCENT, _ACCENT2, _WARN]
    bars2 = ax2.bar(conf_labels, conf_values, color=bar_colors, width=0.5, edgecolor="none")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    for bar, val in zip(bars2, conf_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2%}", ha="center", va="bottom", color=_TEXT, fontsize=9, fontweight="bold")
    ax2.set_title("Prediction Confidence", fontsize=10, fontweight="bold", pad=10)
    _apply_chart_style(ax2, fig)

    # ── Panel 3: Segmentation scores (if GT available) ──
    if has_seg:
        ax3 = axes[2]
        score_labels = ["aAcc", "mAcc", "mIoU"]
        score_values = [seg_metrics.get(k, 0) for k in score_labels]
        bars3 = ax3.bar(score_labels, score_values, color=[_ACCENT, "#06b6d4", "#8b5cf6"],
                        width=0.5, edgecolor="none")
        ax3.set_ylim(0, 1.05)
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        for bar, val in zip(bars3, score_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2%}", ha="center", va="bottom", color=_TEXT, fontsize=9, fontweight="bold")
        ax3.set_title("Segmentation Quality", fontsize=10, fontweight="bold", pad=10)
        _apply_chart_style(ax3, fig)

    fig.tight_layout(pad=1.5)
    return fig


def _build_timing_html(timing: dict) -> str:
    """Build a styled HTML breakdown of timing info."""
    rows = ""
    pipeline_keys = [
        ("load_ms", "Image Load", "\U0001f4c2"),
        ("preprocess_ms", "Preprocess", "\u2699\ufe0f"),
        ("inference_ms", "Inference", "\U0001f9e0"),
        ("postprocess_ms", "Postprocess", "\U0001f3a8"),
        ("total_ms", "Total", "\u23f1\ufe0f"),
    ]
    # Video keys
    video_keys = [
        ("processed_frames", "Processed Frames", "\U0001f39e\ufe0f"),
        ("avg_inference_ms_per_frame", "Avg Inference/Frame", "\U0001f9e0"),
        ("avg_total_ms_per_frame", "Avg Total/Frame", "\u23f1\ufe0f"),
        ("effective_fps", "Effective FPS", "\U0001f680"),
        ("total_video_runtime_ms", "Total Runtime", "\u23f3"),
    ]
    keys_to_use = pipeline_keys if "total_ms" in timing else video_keys

    total_ms = timing.get("total_ms", timing.get("total_video_runtime_ms", 1))

    for key, label, icon in keys_to_use:
        value = timing.get(key)
        if value is None:
            continue
        is_total = key in ("total_ms", "total_video_runtime_ms")
        # Build a mini progress bar for time-based keys
        if key not in ("processed_frames", "effective_fps", "image_width", "image_height") and total_ms > 0:
            pct = min((value / total_ms) * 100, 100) if not is_total else 100
            bar_color = _ACCENT if not is_total else "#8b5cf6"
            bar_html = f'<div style="background:rgba(148,163,184,0.1);border-radius:4px;height:6px;margin-top:4px;"><div style="width:{pct:.0f}%;background:{bar_color};height:6px;border-radius:4px;"></div></div>'
        else:
            bar_html = ""

        # Format value
        if key == "effective_fps":
            val_text = f"{value:.1f} fps"
        elif key == "processed_frames":
            val_text = f"{int(value)} frames"
        else:
            val_text = f"{value:.1f} ms"

        font_weight = "700" if is_total else "400"
        border_top = "border-top:1px solid #334155;" if is_total else ""
        rows += f'''
        <tr style="{border_top}">
            <td style="padding:6px 10px;color:{_SUBTEXT};font-size:13px;">{icon}</td>
            <td style="padding:6px 8px;color:{_TEXT};font-size:13px;font-weight:{font_weight};">{label}</td>
            <td style="padding:6px 10px;color:{_ACCENT2};font-size:13px;font-weight:600;text-align:right;">{val_text}{bar_html}</td>
        </tr>'''

    # Resolution row if available
    if "image_width" in timing:
        rows += f'''
        <tr>
            <td style="padding:6px 10px;color:{_SUBTEXT};font-size:13px;">\U0001f4d0</td>
            <td style="padding:6px 8px;color:{_TEXT};font-size:13px;">Resolution</td>
            <td style="padding:6px 10px;color:{_ACCENT2};font-size:13px;font-weight:600;text-align:right;">{timing["image_width"]}\u00d7{timing["image_height"]}</td>
        </tr>'''

    return f'''
    <div style="background:{_CARD_BG};border:1px solid #334155;border-radius:12px;overflow:hidden;">
        <div style="background:rgba(16,185,129,0.08);padding:8px 14px;border-bottom:1px solid #334155;">
            <span style="color:{_ACCENT2};font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:0.05em;">\u23f1 Timing Breakdown</span>
        </div>
        <table style="width:100%;border-collapse:collapse;">
            {rows}
        </table>
    </div>'''


def _build_video_metrics_chart(metrics: dict) -> plt.Figure:
    """Build a metrics chart for video inference results."""
    prediction = metrics.get("prediction", {})
    top_classes = prediction.get("top_classes", [])
    mean_conf = prediction.get("mean_frame_confidence", 0)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Panel 1: Top classes
    ax = axes[0]
    if top_classes:
        names = [c["class_name"][:18] for c in reversed(top_classes[:8])]
        ratios = [c["ratio"] * 100 for c in reversed(top_classes[:8])]
        colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(names))]
        bars = ax.barh(names, ratios, color=colors, height=0.6, edgecolor="none")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel("Pixel %")
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", color=_SUBTEXT, fontsize=7)
    ax.set_title("Top Classes (all frames)", fontsize=10, fontweight="bold", pad=10)
    _apply_chart_style(ax, fig)

    # Panel 2: Mean confidence
    ax2 = axes[1]
    bars2 = ax2.bar(["Mean Frame\nConfidence"], [mean_conf], color=[_ACCENT], width=0.4, edgecolor="none")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax2.text(0, mean_conf + 0.03, f"{mean_conf:.2%}", ha="center", va="bottom",
             color=_TEXT, fontsize=12, fontweight="bold")
    ax2.set_title("Avg Confidence", fontsize=10, fontweight="bold", pad=10)
    _apply_chart_style(ax2, fig)

    fig.tight_layout(pad=1.5)
    return fig


APP_CSS = """
/* ── Global ── */
.gradio-container {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
  max-width: 1100px !important;
  margin: 0 auto;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Hero banner ── */
#hero {
  border: 1px solid rgba(99, 200, 160, 0.25);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  padding: 20px 28px;
  margin-bottom: 8px;
}
#hero h1 {
  color: #ecfdf5 !important;
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  margin-bottom: 4px !important;
}
#hero p, #hero li, #hero span {
  color: #a7f3d0 !important;
  font-size: 0.88rem !important;
  line-height: 1.45 !important;
}
#hero code {
  background: rgba(16, 185, 129, 0.2) !important;
  color: #6ee7b7 !important;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.82rem !important;
}
#model-info p, #model-info li, #model-info span, #model-info strong {
  color: #d1fae5 !important;
}

/* ── Cards / panels ── */
.panel, .panel > div {
  border: 1px solid rgba(148, 163, 184, 0.15) !important;
  border-radius: 12px !important;
  background: rgba(30, 41, 59, 0.6) !important;
}

/* ── Buttons ── */
.primary {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
  transition: all 0.2s ease !important;
}
.primary:hover {
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
  transform: translateY(-1px);
}

/* ── Tabs ── */
.tab-nav button {
  color: #94a3b8 !important;
  font-weight: 500 !important;
  border: none !important;
  transition: color 0.2s ease;
}
.tab-nav button.selected {
  color: #6ee7b7 !important;
  border-bottom: 2px solid #10b981 !important;
}

/* ── Inputs ── */
.block .wrap, .block textarea, .block input {
  background: rgba(15, 23, 42, 0.7) !important;
  border: 1px solid rgba(148, 163, 184, 0.2) !important;
  border-radius: 8px !important;
  color: #e2e8f0 !important;
}
label span {
  color: #cbd5e1 !important;
  font-size: 0.82rem !important;
}
.block .info {
  color: #64748b !important;
  font-size: 0.75rem !important;
}

/* ── Compact spacing ── */
.gap { gap: 8px !important; }
.block { margin-bottom: 4px !important; }

/* ── Full-width file upload drop zone ── */
.block .upload-button, .block .file-preview {
  width: 100% !important;
  max-width: 100% !important;
}
.block .wrap.svelte-1ipelgc,
.block .upload-button .wrap {
  width: 100% !important;
  max-width: 100% !important;
}
"""


def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks demo for FoodSeg103.

    Sets up the full UI layout — hero banner, model selector, advanced
    options, image/video upload tabs, output panels, metrics chart, and
    timing HTML — and wires the *Run Inference* button to
    :class:`~app.service.FoodSegDemoService`.

    Returns:
        gr.Blocks: A ready-to-launch Gradio application instance.
    """
    service = FoodSegDemoService()
    model_labels = service.get_model_labels()
    default_model = service.get_default_model_label()
    test_images = service.get_test_image_choices()
    default_test_image = service.get_default_test_image()

    with gr.Blocks(
        title="FoodSeg103 Gradio Demo",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="emerald",
            neutral_hue="slate",
        ),
        css=APP_CSS,
    ) as demo:
        # ── Hero ──
        with gr.Column(elem_id="hero"):
            gr.Markdown(
                "# \U0001f371 FoodSeg103 Demo\n"
                "Semantic segmentation on food images & videos \u2014 switch models, "
                "pick test images, or upload your own."
            )
            model_info = gr.Markdown(
                service.describe_model(default_model),
                elem_id="model-info",
            )

        # ── Controls row ──
        with gr.Row(equal_height=True):
            model_dropdown = gr.Dropdown(
                choices=model_labels,
                value=default_model,
                label="Model type",
                scale=2,
            )
            alpha_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                label="Overlay alpha",
                scale=3,
            )

        model_dropdown.change(service.describe_model, inputs=model_dropdown, outputs=model_info)

        # ── Advanced options ──
        with gr.Accordion("Advanced options", open=False):
            with gr.Row(equal_height=True):
                selected_test_image = gr.Dropdown(
                    choices=test_images,
                    value=default_test_image,
                    label="Built-in test image",
                    info="Uploaded files take priority over the selected test image.",
                    scale=1,
                )
                checkpoint_override = gr.Textbox(
                    label="Checkpoint override",
                    placeholder="Leave empty for default checkpoint",
                    scale=1,
                )

        # ── Upload area: sub-tabs for Image vs Video ──
        with gr.Tab("\U0001f4f7 Upload Image"):
            uploaded_image = gr.File(
                file_types=["image"], type="filepath",
                label="Upload an image file",
                height=120,
                
            )
        with gr.Tab("\U0001f3ac Upload Video"):
            uploaded_video = gr.Video(
                label="Upload a video file",
                elem_classes=["panel"],
            )

        run_btn = gr.Button("\u25b6 Run Inference", variant="primary", size="lg")

        # ── Output: Image results ──
        with gr.Column(visible=True) as image_output_group:
            with gr.Row(equal_height=True):
                input_preview = gr.Image(
                    label="Input", elem_classes=["panel"], height=280,
                )
                overlay_image = gr.Image(
                    label="Result Overlay", elem_classes=["panel"], height=280,
                )

        # ── Output: Video results ──
        with gr.Column(visible=False) as video_output_group:
            overlay_video = gr.Video(label="Overlay video", elem_classes=["panel"])

        # ── Metrics & Timing (shared) ──
        metrics_plot = gr.Plot(label="\U0001f4ca Metrics")
        timing_html = gr.HTML()

        # ── Inference logic ──
        def _run_inference(
            model_label, sel_test_img, img_file, video_file,
            ckpt_override, alpha,
        ):
            has_video = video_file is not None

            if has_video:
                # ── Video mode ──
                output_path, metrics, timing = service.process_video(
                    model_label, video_file, ckpt_override, alpha,
                )
                fig = _build_video_metrics_chart(metrics)
                html = _build_timing_html(timing)
                return (
                    gr.update(visible=False),   # image_output_group
                    gr.update(visible=True),    # video_output_group
                    None,                       # input_preview
                    None,                       # overlay_image
                    output_path,                # overlay_video
                    fig,                        # metrics_plot
                    html,                       # timing_html
                )
            else:
                # ── Image mode (default) ──
                img_in, _gt, overlay, metrics, timing = service.predict_image(
                    model_label, sel_test_img, img_file, None, ckpt_override, alpha,
                )
                fig = _build_metrics_chart(metrics)
                html = _build_timing_html(timing)
                return (
                    gr.update(visible=True),    # image_output_group
                    gr.update(visible=False),   # video_output_group
                    img_in,                     # input_preview
                    overlay,                    # overlay_image
                    None,                       # overlay_video
                    fig,                        # metrics_plot
                    html,                       # timing_html
                )

        run_btn.click(
            fn=_run_inference,
            inputs=[
                model_dropdown,
                selected_test_image,
                uploaded_image,
                uploaded_video,
                checkpoint_override,
                alpha_slider,
            ],
            outputs=[
                image_output_group,
                video_output_group,
                input_preview,
                overlay_image,
                overlay_video,
                metrics_plot,
                timing_html,
            ],
        )

        demo.queue(api_open=False)

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Gradio demo launcher.

    Returns:
        argparse.Namespace: Parsed arguments with ``host``, ``port``,
            and ``share`` fields.
    """
    parser = argparse.ArgumentParser(description="Launch the FoodSeg103 Gradio demo.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio sharing.")
    return parser.parse_args()


def main() -> None:
    """Parse arguments, build the Gradio demo, and launch the server."""
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
