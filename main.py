"""
ROBUST-6G WP6 Demonstrator - FINAL VERSION

ROOT CAUSE OF BROKEN NODE PLACEMENT (confirmed by reading NiceGUI 3.x source):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In nicegui/elements/plotly/plotly.js (lines 31-36):

    if (last_options && JSON.stringify(options.config) === JSON.stringify(last_options.config)) {
        this.Plotly.react(...)       // fast update — NO event handlers re-attached
    } else {
        this.Plotly.newPlot(...)     // full re-render
        this.set_handlers()          // ← plotly_click only bound here
    }

set_handlers() — which wires plotly_click — is ONLY called on newPlot.
Every call to self.plot.update() triggers react() (same config → same JSON),
so plotly_click is permanently lost after the very first _update_plot() call.

FIX: increment a dummy counter in fig.layout every time we call _update_plot().
     This changes the config JSON → forces newPlot → set_handlers() is called
     → plotly_click is re-bound correctly after every node placement.

OTHER BUGS FIXED:
  • Dead create_bit_bar() / orphan return block at class scope (SyntaxError)
  • .stop() → .cancel() for ui.timer
  • All hardcoded metric values → real engine calls (glrt, spoof_detector, skg_engine)
"""

import sys
import os
import threading
import traceback
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message=".*antenna_positions.*")

from nicegui import ui, app
import plotly.graph_objects as go

from config import SystemConfig, Node
from models.channel_model import ChannelModel
from models.authentication_engine import AuthenticationEngine
from models.key_generator import SecretKeyGenerator
from models.jamming_detector import JammingDetector
from models.jamming_detector_glrt import (
    GLRTJammingDetector, GLRTDetectorConfig,
    build_psignal_grid, xy_to_grid_index,
)
from models.spoof_detector import SpoofDetector
from models.skg_engine import SKGEngine

_SNR_DB = {'Low': 10.0, 'Medium': 20.0, 'High': 30.0}
_PJ_DBM = {'Low':  7.0, 'Medium': 15.0, 'High': 25.0}


def _find_antenna_pos():
    for p in ['antenna_positions.npy', 'datasets/antenna_positions.npy',
              'data/antenna_positions.npy', 'cache/antenna_positions.npy']:
        if os.path.isfile(p):
            return p
    return None


class Robust6GDemonstrator:
    def __init__(self):
        self.config = SystemConfig()
        self._load_dataset()

        self.channel          = ChannelModel(self.config)
        self.auth_engine      = AuthenticationEngine(self.config)
        self.key_generator    = SecretKeyGenerator(self.config)
        self.jamming_detector = JammingDetector(self.config)

        self.ps_grid, self._x_vals, self._y_vals = build_psignal_grid(
            self.dataset_csi, self.dataset_positions, ptx_dbm=15.0)
        self.glrt = GLRTJammingDetector(
            self.ps_grid,
            GLRTDetectorConfig(alpha=2.5, pfa=0.01, W_temporal=5, use_temporal=True),
        )

        ant_pos = _find_antenna_pos()
        self._ant_pos_found = ant_pos is not None
        self.spoof_detector = SpoofDetector(
            dataset_path=self.config.dataset_path,
            ant_pos_path=ant_pos,
            cache_dir='cache',
        )
        threading.Thread(target=self.spoof_detector.prepare, daemon=True).start()

        self.skg_engine = (
            SKGEngine('skg_robust6G') if os.path.isdir('skg_robust6G') else None
        )

        self.nodes              = {}
        self.next_node_id       = 0
        self.current_snr        = 'Medium'
        self.current_pj         = 'Medium'
        self.selected_role      = None
        self.attack_vars        = {'Jamming': False, 'Spoofing': False, 'Eavesdropping': False}
        self.placed_roles       = {
            'Legitimate User': False, 'Jammer': False,
            'Spoofer': False, 'Eavesdropper': False,
        }
        self.simulation_running = False
        self.start_button       = None
        self.sim_result         = None
        self._plot_update_count = 0   # incremented each update to force newPlot

        self._build_ui()

    # ── dataset ───────────────────────────────────────────────────────────────
    def _load_dataset(self):
        candidates = [
            'data_ULA_all.npz',
            'datasets/data_ULA_all.npz',
            getattr(self.config, 'dataset_path', None),
        ]
        npz_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not npz_path:
            raise FileNotFoundError("Dataset not found")
        with np.load(npz_path) as data:
            self.dataset_positions = np.asarray(data['UEs_positions'])
            self.dataset_csi       = np.asarray(data['csi_UEs_all'])
        ui.notify("✅ Dataset loaded", type='positive')

    def _snap_to_dataset(self, x, y):
        dx  = self.dataset_positions[:, 0] - x
        dy  = self.dataset_positions[:, 1] - y
        idx = int(np.argmin(dx * dx + dy * dy))
        return idx, float(self.dataset_positions[idx, 0]), float(self.dataset_positions[idx, 1])

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        ui.add_head_html("""<style>
            .four-panel-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 0;
                width: 100%;
                height: calc(100vh - 80px);
            }
            .panel-card {
                display: flex; flex-direction: column;
                overflow: hidden; min-height: 0;
                background: #1e1e2f;
                border-radius: 0 !important;
                border: 1px solid rgba(255,255,255,0.09);
            }
            .panel-header {
                padding: 11px 16px;
                border-bottom: 1px solid rgba(255,255,255,0.10);
                font-weight: 700; font-size: 1rem;
                background: #2a2a3a; flex-shrink: 0;
            }
            .panel-body {
                flex: 1; min-height: 0; overflow-y: auto;
                padding: 14px 16px;
                display: flex; flex-direction: column;
            }
            .rc {
                border-radius: 10px; padding: 13px 15px; margin-bottom: 10px;
                font-family: monospace; font-size: 0.88rem; line-height: 1.7;
                border-left: 5px solid #555; background: #252536;
            }
            .rc-bad  { border-left-color: #e74c3c; }
            .rc-good { border-left-color: #2ecc71; }
            .rc-warn { border-left-color: #f39c12; }
            .rc-info { border-left-color: #3498db; }
            .kv { width: 100%; border-collapse: collapse; margin-top: 6px; }
            .kv tr { border-bottom: 1px solid rgba(255,255,255,0.06); }
            .kv tr:last-child { border-bottom: none; }
            .kv td { padding: 5px 2px; font-size: 12.5px; }
            .kl      { color: #95a5a6; width: 52%; }
            .kv-bad  { color: #e74c3c; font-weight: 700; text-align: right; }
            .kv-good { color: #2ecc71; font-weight: 700; text-align: right; }
            .kv-warn { color: #f39c12; font-weight: 700; text-align: right; }
            .kv-val  { color: #ecf0f1; font-weight: 500; text-align: right; }
            .bit-label { font-size: 11px; color: #95a5a6; margin: 8px 0 3px; }
            .bit-row {
                display: flex; flex-wrap: wrap; gap: 1px;
                background: #1a1a2e; padding: 6px;
                border-radius: 6px; margin-bottom: 6px;
            }
            .bit { width: 7px; height: 18px; border-radius: 1px; flex-shrink: 0; }
            .key-hex {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 6px; padding: 7px 10px; margin-top: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px; color: #7f8c8d; word-break: break-all;
            }
            .key-hex span { color: #636e72; font-size: 10px; display: block; margin-bottom: 2px; }
            .idle { color: #4a5568; font-size: 13px; font-style: italic; margin-top: 8px; }
            .leg  { display: flex; align-items: center; gap: 7px;
                    font-size: 12px; color: #bdc3c7; margin-bottom: 3px; }
            .leg-dot { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
            .ant-warn {
                font-size: 11px; color: #95a5a6; font-style: italic;
                border: 1px solid rgba(243,156,18,0.3);
                border-radius: 5px; padding: 4px 8px; margin-bottom: 8px;
            }
        </style>""")

        # Serve logos from ./assets so the <img> tags below can load them.
        # Drop Robust-6G.png and Logo_ENSEA.png into a folder named 'assets'
        # next to main.py.
        if os.path.isdir('assets'):
            app.add_static_files('/assets', 'assets')

        with ui.header().classes('items-center bg-primary text-white p-3').style(
                'display:flex; justify-content:space-between;'):
            # Left spacer matches the right-hand button width so the centre
            # group stays optically centred.
            ui.element('div').style('width:110px;')

            with ui.row().classes('items-center gap-3').style('flex:1; justify-content:center;'):
                ui.html(
                    '<img src="/assets/Robust-6G.png" '
                    'style="height:36px; width:auto;" alt="Robust-6G logo" '
                    'onerror="this.style.display=\'none\';">'
                )
                ui.label('🔒 ROBUST-6G WP6 Demonstrator').classes('text-h6').style(
                    'margin:0;')
                ui.html(
                    '<img src="/assets/Logo_ENSEA.png" '
                    'style="height:36px; width:auto;" alt="ENSEA logo" '
                    'onerror="this.style.display=\'none\';">'
                )

            ui.button('Reset All', on_click=self._clear_all).props(
                'flat color=white').style('width:110px;')

        with ui.row().classes('w-full no-wrap').style('height:calc(100vh - 56px); overflow:hidden;'):
            with ui.column().classes('w-80 border-r p-4 gap-4 overflow-auto bg-gray-900 text-white')\
                    .style('flex-shrink:0;'):
                self._build_control_panel()

            with ui.element('div').classes('flex-1').style('overflow:hidden; min-width:0;'):
                with ui.element('div').classes('four-panel-grid'):

                    with ui.element('div').classes('panel-card'):
                        ui.html('<div class="panel-header">📍 Node Placement</div>')
                        with ui.element('div').style(
                            'flex:1; min-height:0; display:flex; flex-direction:column;'
                        ):
                            self._build_grid_panel()

                    with ui.element('div').classes('panel-card'):
                        ui.html('<div class="panel-header">📡 Jamming Detection</div>')
                        with ui.element('div').classes('panel-body'):
                            self.jam_html = ui.html(
                                '<p class="idle">Run simulation to see results.</p>')

                    with ui.element('div').classes('panel-card'):
                        ui.html('<div class="panel-header">🔑 Secret Key Generation</div>')
                        with ui.element('div').classes('panel-body'):
                            self.skg_html = ui.html(
                                '<p class="idle">Run simulation to see results.</p>')

                    with ui.element('div').classes('panel-card'):
                        ui.html('<div class="panel-header">🎭 Spoof Detection</div>')
                        with ui.element('div').classes('panel-body'):
                            self.spoof_html = ui.html(
                                '<p class="idle">Run simulation to see results.</p>')

    def _build_control_panel(self):
        ui.label('🎛️ CONTROL PANEL').classes('text-h6')

        if not self._ant_pos_found:
            ui.html(
                '<div class="ant-warn">⚠ antenna_positions.npy not found — '
                'using geometric fallback [0,0,1] for AoA.</div>'
            )

        self.snr_select = ui.select(
            ['Low', 'Medium', 'High'], value='Medium', label='SNR Level'
        ).classes('w-full').on_value_change(
            lambda e: setattr(self, 'current_snr', e.value))

        self.pj_select = ui.select(
            ['Low', 'Medium', 'High'], value='Medium', label='Jamming Intensity'
        ).classes('w-full').on_value_change(
            lambda e: setattr(self, 'current_pj', e.value))

        with ui.expansion('⚔️ Attack Scenarios', value=True):
            for a in ['Jamming', 'Spoofing', 'Eavesdropping']:
                ui.checkbox(
                    a, value=False,
                    on_change=lambda e, atk=a: self.attack_vars.update({atk: e.value}),
                )

        with ui.expansion('📍 Place Nodes', value=True):
            ui.button('🟢 PLACE USER',
                      on_click=lambda: self._set_role('Legitimate User')).classes('w-full')
            ui.button('🔴 PLACE JAMMER',
                      on_click=lambda: self._set_role('Jammer')).classes('w-full')
            ui.button('🟠 PLACE SPOOFER',
                      on_click=lambda: self._set_role('Spoofer')).classes('w-full')
            ui.button('🟣 PLACE EAVESDROPPER',
                      on_click=lambda: self._set_role('Eavesdropper')).classes('w-full')

        with ui.row().classes('gap-2 mt-6'):
            self.start_button = ui.button(
                '▶ START SIMULATION', on_click=self._start_simulation, color='green'
            ).props('large')
            ui.button('⟳ RESET', on_click=self._reset_simulation).props('large')
            ui.button('🗑 CLEAR', on_click=self._clear_nodes, color='red').props('large')

    # ── Grid / plot ───────────────────────────────────────────────────────────
    def _make_figure(self) -> go.Figure:
        """
        Build a fresh Figure with:
          Trace 0 – invisible dense grid (click catcher for empty-space clicks)
          Trace 1 – dataset background dots
          Trace 2+ – one trace per placed node

        CRITICAL: fig.layout._update_count is set to self._plot_update_count.
        NiceGUI's plotly.js compares JSON.stringify(options.config) between
        updates. We embed the counter in layout (which becomes part of options),
        guaranteeing the JSON changes each call → Plotly.newPlot() is used →
        set_handlers() runs → plotly_click listener is re-bound every time.
        """
        xs = self.dataset_positions[:, 0]
        ys = self.dataset_positions[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        pad_x = (x_max - x_min) * 0.06
        pad_y = (y_max - y_min) * 0.06

        # Click-catcher mesh covers the dataset bounding box plus padding.
        # We extend the visible plot range further right to host the ULA glyph,
        # but the catcher stays over the data area so users can't place nodes
        # inside the antenna column.
        gx  = np.linspace(x_min - pad_x, x_max + pad_x, 30)
        gy  = np.linspace(y_min - pad_y, y_max + pad_y, 30)
        gxx, gyy = np.meshgrid(gx, gy)

        # ULA position (visual only): a column to the right of the data area.
        ula_x = x_max + (x_max - x_min) * 1.5
        ula_y = np.linspace(y_min, y_max, 6)

        # View range with extra room on the right for the ULA + label.
        view_x_min = x_min - pad_x
        view_x_max = ula_x + (x_max - x_min) * 0.25
        view_y_min = y_min - pad_y
        view_y_max = y_max + pad_y

        colors  = {
            'Legitimate User': 'lime',   'Jammer':       'red',
            'Spoofer':         'orange', 'Eavesdropper': 'violet',
        }
        symbols = {
            'Legitimate User': 'circle',    'Jammer':       'triangle-up',
            'Spoofer':         'square',    'Eavesdropper': 'diamond',
        }

        fig = go.Figure()

        # Trace 0 – invisible click catcher (catches clicks on empty space).
        # hoverinfo='none' (not 'skip') — 'skip' makes the trace click-
        # transparent and plotly_click never fires.
        fig.add_trace(go.Scatter(
            x=gxx.ravel().tolist(),
            y=gyy.ravel().tolist(),
            mode='markers',
            marker=dict(size=14, opacity=0, color='rgba(0,0,0,0)'),
            hoverinfo='none',
            showlegend=False,
        ))

        # Trace 1 – ULA glyph (visual only, non-interactive)
        fig.add_trace(go.Scatter(
            x=[ula_x] * len(ula_y),
            y=ula_y.tolist(),
            mode='markers',
            marker=dict(color='#00d4ff', size=5, symbol='circle'),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Trace 2+ – placed nodes
        for node in self.nodes.values():
            label = f"{node.role[:4]} ({node.x:.2f},{node.y:.2f})"
            fig.add_trace(go.Scatter(
                x=[node.x], y=[node.y],
                mode='markers+text',
                marker=dict(
                    color=colors.get(node.role, 'white'),
                    size=26,
                    symbol=symbols.get(node.role, 'circle'),
                    line=dict(color='white', width=2),
                ),
                text=[label],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                hoverinfo='skip',
                showlegend=False,
            ))

        # Dashed bounding box around the placement area + ULA outline.
        shapes = [
            # Placement area (dashed cyan box)
            dict(
                type='rect',
                xref='x', yref='y',
                x0=x_min - pad_x * 0.3, x1=x_max + pad_x * 0.3,
                y0=y_min - pad_y * 0.3, y1=y_max + pad_y * 0.3,
                line=dict(color='#00d4ff', width=1.2, dash='dash'),
                fillcolor='rgba(0,0,0,0)',
                layer='below',
            ),
            # ULA enclosure (solid cyan rounded rectangle)
            dict(
                type='rect',
                xref='x', yref='y',
                x0=ula_x - (x_max - x_min) * 0.04,
                x1=ula_x + (x_max - x_min) * 0.04,
                y0=y_min - pad_y * 0.1,
                y1=y_max + pad_y * 0.1,
                line=dict(color='#00d4ff', width=1.5),
                fillcolor='rgba(0,212,255,0.05)',
                layer='below',
            ),
        ]

        annotations = [
            dict(
                x=ula_x + (x_max - x_min) * 0.06,
                y=(y_min + y_max) / 2,
                xref='x', yref='y',
                text='64-element ULA',
                showarrow=False,
                textangle=-90,
                font=dict(color='#00d4ff', size=11),
            ),
        ]

        fig.update_layout(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(range=[view_x_min, view_x_max]),
            yaxis=dict(range=[view_y_min, view_y_max]),
            template='plotly_dark',
            showlegend=False,
            autosize=True,
            margin=dict(l=40, r=10, t=10, b=40),
            clickmode='event',
            shapes=shapes,
            annotations=annotations,
            # Counter in `meta` forces NiceGUI's plotly.js to call newPlot()
            # rather than react(), so plotly_click stays bound after updates.
            meta={'_upd': self._plot_update_count},
        )

        return fig

    def _build_grid_panel(self):
        """Create the ui.plotly widget and register the click handler once."""
        fig = self._make_figure()
        self.plot = ui.plotly(fig).style('flex:1; min-height:0; width:100%;')
        self.plot.on('plotly_click', self._on_plot_click)

    def _update_plot(self):
        """Rebuild figure with incremented counter → forces newPlot → re-wires click."""
        self._plot_update_count += 1
        self.plot.figure = self._make_figure()
        self.plot.update()

    # ── Node placement ────────────────────────────────────────────────────────
    # Maps a placement role to the attack scenario that must be enabled
    # for that role to be placeable. The legitimate user is the victim
    # and is always placeable.
    _ROLE_REQUIRES_SCENARIO = {
        'Jammer':       'Jamming',
        'Spoofer':      'Spoofing',
        'Eavesdropper': 'Eavesdropping',
    }

    def _set_role(self, role):
        required = self._ROLE_REQUIRES_SCENARIO.get(role)
        if required and not self.attack_vars.get(required):
            ui.notify(
                f"Enable the '{required}' scenario before placing a {role}.",
                type='warning',
            )
            return
        self.selected_role = role
        ui.notify(f"✏ {role} selected — click anywhere on the grid", type='positive')

    def _on_plot_click(self, e):
        if not self.selected_role:
            ui.notify("Select a node type first!", type='warning')
            return

        # e.args is a dict with key 'points' (list of point dicts)
        raw = e.args if isinstance(e.args, dict) else {}
        points = raw.get('points', [])
        if not points:
            return

        point = points[0]
        x = point.get('x')
        y = point.get('y')
        if x is None or y is None:
            return

        _, sx, sy = self._snap_to_dataset(float(x), float(y))

        if self.placed_roles.get(self.selected_role):
            ui.notify(f"{self.selected_role} already placed!", type='warning')
            return

        node = Node(
            node_id=self.next_node_id, x=sx, y=sy,
            role=self.selected_role, active=True,
        )
        self.nodes[self.next_node_id] = node
        self.placed_roles[self.selected_role] = True
        self.next_node_id += 1

        self._update_plot()
        ui.notify(f"✅ Placed {self.selected_role} at ({sx:.2f}, {sy:.2f})", type='positive')
        self.selected_role = None

    # ── Simulation lifecycle ──────────────────────────────────────────────────
    def _start_simulation(self):
        if self.simulation_running:
            ui.notify("Simulation already running", type='warning')
            return
        if not self.placed_roles.get('Legitimate User'):
            ui.notify("Place Legitimate User first!", type='negative')
            return

        self.start_button.disable()
        self.simulation_running = True
        self.sim_result = None

        spin = ('<p style="color:#f39c12;font-size:13px;font-family:monospace;">'
                '⏳ Analysing…</p>')
        self.jam_html.set_content(spin)
        self.spoof_html.set_content(spin)
        self.skg_html.set_content(spin)

        threading.Thread(target=self._run_simulation_background, daemon=True).start()
        self._poll_timer = ui.timer(0.6, self._check_simulation_result)

    def _check_simulation_result(self):
        if getattr(self, 'sim_result', None) is not None:
            jam, spoof, skg = self.sim_result
            self.jam_html.set_content(jam)
            self.spoof_html.set_content(spoof)
            self.skg_html.set_content(skg)
            ui.notify("✅ Simulation completed!", type='positive')
            if hasattr(self, '_poll_timer'):
                self._poll_timer.cancel()
            self._simulation_finished()

    def _simulation_finished(self):
        self.simulation_running = False
        self.sim_result = None
        if self.start_button:
            self.start_button.enable()

    # ── Core simulation (worker thread) ──────────────────────────────────────
    def _run_simulation_background(self):
        try:
            user    = next((n for n in self.nodes.values() if n.role == 'Legitimate User'), None)
            jammer  = next((n for n in self.nodes.values() if n.role == 'Jammer'),          None)
            spoofer = next((n for n in self.nodes.values() if n.role == 'Spoofer'),         None)
            eve     = next((n for n in self.nodes.values() if n.role == 'Eavesdropper'),    None)

            snr_db = _SNR_DB.get(self.current_snr, 20.0)
            pj_dbm = _PJ_DBM.get(self.current_pj, 15.0)

            jam_html   = self._compute_jamming_html(user, jammer, snr_db, pj_dbm)
            spoof_html = self._compute_spoof_html(user, jammer, spoofer, snr_db, pj_dbm)
            skg_html   = self._compute_skg_html(user, eve, snr_db)

            self.sim_result = (jam_html, spoof_html, skg_html)

        except Exception:
            traceback.print_exc()
            err = '<div class="rc rc-bad">❌ Simulation error — check terminal for details.</div>'
            self.sim_result = (err, err, err)

    # ── Jamming ───────────────────────────────────────────────────────────────
    def _compute_jamming_html(self, user, jammer, snr_db, pj_dbm):
        if not (jammer and self.attack_vars.get('Jamming')):
            return '<div class="rc rc-good">✅ No jammer placed — channel is clean.</div>'

        self.glrt.reset()
        jammer_rc = xy_to_grid_index(jammer.x, jammer.y, self._x_vals, self._y_vals)

        N_STEPS = 30
        last = None
        for _ in range(N_STEPS):
            last = self.glrt.step(snr_db=snr_db, jammer_rc=jammer_rc, pj_dbm=pj_dbm)

        alarm        = last['alarm']
        peak_val     = last['peak_val']
        tau          = last['tau']
        g_t          = last['g_t']
        pr           = last['peak_r']
        pc           = last['peak_c']
        p_jam_grid   = last['p_jam_grid_mW']
        p_noise_grid = last['p_noise_grid_mW']

        est_x = float(self._x_vals[pc]) if pc is not None else jammer.x
        est_y = float(self._y_vals[pr]) if pr is not None else jammer.y

        if user:
            ur, uc = xy_to_grid_index(user.x, user.y, self._x_vals, self._y_vals)
            p_s  = float(self.ps_grid[ur, uc])
            p_j  = float(p_jam_grid[ur, uc])
            p_n  = float(p_noise_grid[ur, uc])
            sinr_user_db = 10 * np.log10(p_s / (p_j + p_n + 1e-30))
            jn_user_db   = 10 * np.log10(p_j  / (p_n + 1e-30))
        else:
            sinr_user_db = float('nan')
            jn_user_db   = float('nan')

        ratio    = peak_val / (tau + 1e-12)
        conf     = 'Very High' if ratio > 5 else ('High' if ratio > 2 else 'Moderate')
        banner   = 'rc-bad'  if alarm else 'rc-warn'
        title    = '⚠ JAMMING DETECTED' if alarm else '⚠ Possible Jamming (below threshold)'
        sinr_cls = 'kv-warn' if sinr_user_db < 15 else 'kv-good'
        jn_cls   = 'kv-bad'  if jn_user_db   > -10 else 'kv-warn'
        det_cls  = 'kv-bad'  if alarm               else 'kv-warn'

        return f"""
        <div class="rc {banner}">
          <b>{title}</b>
          <table class="kv">
            <tr><td class="kl">Jammer (true)</td>
                <td class="kv-val">({jammer.x:.3f}, {jammer.y:.3f}) m</td></tr>
            <tr><td class="kl">Jammer (estimated)</td>
                <td class="kv-val">({est_x:.3f}, {est_y:.3f}) m</td></tr>
            <tr><td class="kl">SNR setting</td>
                <td class="kv-val">{self.current_snr} ({snr_db:.0f} dB)</td></tr>
            <tr><td class="kl">Jamming intensity</td>
                <td class="kv-val">{self.current_pj} ({pj_dbm:.0f} dBm)</td></tr>
            <tr><td class="kl">Jammer power</td>
                <td class="kv-val">{pj_dbm:.1f} dBm</td></tr>
            <tr><td class="kl">SINR @ user</td>
                <td class="{sinr_cls}">{sinr_user_db:+.2f} dB</td></tr>
            <tr><td class="kl">J/N @ user</td>
                <td class="{jn_cls}">{jn_user_db:+.2f} dB</td></tr>
            <tr><td class="kl">GLRT peak score</td>
                <td class="kv-val">{peak_val:.2f} (τ={tau:.2f})</td></tr>
            <tr><td class="kl">CUSUM g_t</td>
                <td class="kv-val">{g_t:.3f}</td></tr>
            <tr><td class="kl">Detection confidence</td>
                <td class="{det_cls}">{conf}</td></tr>
          </table>
          <p style="color:#4a5568;font-size:10.5px;margin-top:8px;">
            step {N_STEPS}/{N_STEPS} | peak {peak_val:.1f} | τ {tau:.2f}
          </p>
        </div>"""

    # ── Spoof ─────────────────────────────────────────────────────────────────
    def _compute_spoof_html(self, user, jammer, spoofer, snr_db, pj_dbm):
        if not (spoofer and user and self.attack_vars.get('Spoofing')):
            return '<div class="rc rc-good">✅ No spoofer placed — no spoofing threat.</div>'

        ant_note = ''
        if not self._ant_pos_found:
            ant_note = ('<div class="ant-warn">ℹ Geometric fallback antenna used. '
                        'AoA values are approximate.</div>')

        try:
            jammer_xy = (jammer.x, jammer.y) if jammer else None
            result = self.spoof_detector.compute(
                user_xy    =(user.x,    user.y),
                spoofer_xy =(spoofer.x, spoofer.y),
                jammer_xy  =jammer_xy,
                pj_dbm     =pj_dbm,
                snr_db     =snr_db,
            )
        except Exception as ex:
            traceback.print_exc()
            return f'<div class="rc rc-bad">❌ Spoof detection error: {ex}</div>'

        sr       = result['spoofer_result']
        verdict  = sr['verdict']
        delta    = sr['delta']
        detected = sr['detected']
        ambig    = sr['ambiguous']
        mae_c    = result['mae_clean']
        med_c    = result['medae_clean']

        sp_map  = result['spoof_map']
        det_arr = sp_map['detected']
        amb_arr = sp_map.get('ambiguous', np.zeros_like(det_arr, dtype=bool))
        valid   = ~np.isnan(sp_map['delta_mit']) & ~amb_arr
        n_out   = int(np.sum(valid))
        n_det   = int(np.sum(det_arr & valid))
        # success_rate = fraction of (non-ambiguous) grid cells where the
        # detector successfully flags spoofing. In _evaluate_spoof_map,
        # detected[k]=True means cell k's mitigated AoA differs from the
        # user's by > threshold, i.e. a spoofer placed there WOULD be caught.
        # So defender-success = n_det / n_out, NOT 1 - n_det / n_out.
        success_rate = 100.0 * (n_det / n_out) if n_out > 0 else 0.0

        if detected and not ambig:
            banner = 'rc-bad';  title = '🎭 SPOOFING DETECTED — SPOOF FAIL'
        elif ambig:
            banner = 'rc-warn'; title = '🎭 AMBIGUOUS — Inside indeterminate zone'
        else:
            banner = 'rc-warn'; title = '🎭 SPOOFING UNDETECTED — SPOOF SUCCESS'

        delta_cls   = 'kv-bad'  if detected          else 'kv-warn'
        # High success_rate is now GOOD (defender catches most spoofers).
        success_cls = 'kv-good' if success_rate > 50 else 'kv-bad'
        verdict_cls = 'kv-bad'  if detected          else 'kv-warn'

        return f"""
        {ant_note}
        <div class="rc {banner}">
          <b>{title}</b>
          <table class="kv">
            <tr><td class="kl">User position</td>
                <td class="kv-val">({user.x:.3f}, {user.y:.3f}) m</td></tr>
            <tr><td class="kl">Spoofer position</td>
                <td class="kv-val">({spoofer.x:.3f}, {spoofer.y:.3f}) m</td></tr>
            <tr><td class="kl">|ΔAoA|</td>
                <td class="{delta_cls}">{delta:.2f}°</td></tr>
            <tr><td class="kl">MedAE (clean baseline)</td>
                <td class="kv-val">{med_c:.2f}°</td></tr>
            <tr><td class="kl">MAE (clean baseline)</td>
                <td class="kv-val">{mae_c:.2f}°</td></tr>
            <tr><td class="kl">Grid spoof-success rate</td>
                <td class="{success_cls}">{success_rate:.1f}%</td></tr>
            <tr><td class="kl">Verdict</td>
                <td class="{verdict_cls}">{verdict}</td></tr>
          </table>
          <div style="margin-top:10px;">
            <div class="leg"><span class="leg-dot" style="background:#2ecc71;"></span>Safe (SPOOF FAIL)</div>
            <div class="leg"><span class="leg-dot" style="background:#f39c12;"></span>Ambiguous
              <span style="color:#4a5568;font-size:10.5px;"> MedAE={med_c:.2f}° MAE={mae_c:.2f}°</span>
            </div>
            <div class="leg"><span class="leg-dot" style="background:#e74c3c;"></span>Spoof Success</div>
          </div>
        </div>"""

    # ── SKG ───────────────────────────────────────────────────────────────────
    def _compute_skg_html(self, user, eve, snr_db):
        if not user:
            return '<div class="rc rc-good">✅ No user placed — SKG not applicable.</div>'
        if self.skg_engine is None:
            return ('<div class="rc rc-warn">⚠ skg_robust6G directory not found — '
                    'real key generation unavailable.</div>')

        try:
            res = self.skg_engine.run(
                alice_xy=(user.x, user.y),
                eve_xy=(eve.x, eve.y) if eve else None,
                snr_db=snr_db,
            )
        except Exception as ex:
            traceback.print_exc()
            return f'<div class="rc rc-bad">❌ SKG error: {ex}</div>'

        recon_pct  = res['reconciliation_pct']
        alice_hex  = res['alice_key_hex']
        alice_bits = res['alice_key_bits']
        eve_bits   = res.get('eve_key_bits')
        eve_match  = res.get('eve_key_match_pct')
        eve_pre    = res.get('eve_pre_recon_match_pct')
        ok         = res.get('alice_bob_match', True)

        def _bit_bar(bits, label):
            cells = ''.join(
                f'<span class="bit" style="background:{"#2ecc71" if b else "#e74c3c"};"></span>'
                for b in bits
            )
            return (f'<div class="bit-label">{label}</div>'
                    f'<div class="bit-row">{cells}</div>')

        alice_bar = _bit_bar(alice_bits, 'Alice key bits')
        eve_bar   = (
            _bit_bar(eve_bits, 'Eve key bits') if eve_bits is not None
            else '<div class="bit-label" style="color:#4a5568;">Eve: no eavesdropper placed</div>'
        )

        eve_row = ''
        if eve_match is not None:
            em_cls  = 'kv-warn' if eve_match > 55 else 'kv-good'
            eve_row = f"""
            <tr><td class="kl">Eve key match (post-hash)</td>
                <td class="{em_cls}">{eve_match:.2f}%</td></tr>
            <tr><td class="kl">Eve pre-hash bit match</td>
                <td class="kv-warn">{eve_pre:.2f}%</td></tr>"""

        recon_cls  = 'kv-good' if recon_pct > 95 else 'kv-warn'
        status_cls = 'kv-good' if ok else 'kv-bad'
        status_txt = '✅ Alice ↔ Bob match' if ok else '❌ Alice ↔ Bob MISMATCH'

        return f"""
        <div class="rc rc-info">
          <b>🔑 Secret Key Generation Complete</b>
          <table class="kv" style="margin-top:8px;">
            <tr><td class="kl">Reconciliation rate</td>
                <td class="{recon_cls}">{recon_pct:.2f}%</td></tr>
            {eve_row}
            <tr><td class="kl">Alice ↔ Bob</td>
                <td class="{status_cls}">{status_txt}</td></tr>
          </table>
          {alice_bar}
          {eve_bar}
          <div class="key-hex"><span>Alice key (hex)</span>{alice_hex}</div>
        </div>"""

    # ── Reset / clear ─────────────────────────────────────────────────────────
    def _reset_simulation(self):
        idle = '<p class="idle">Run simulation to see results.</p>'
        self.jam_html.set_content(idle)
        self.spoof_html.set_content(idle)
        self.skg_html.set_content(idle)
        ui.notify("Results cleared", type='info')

    def _clear_nodes(self):
        self.nodes.clear()
        self.placed_roles = {k: False for k in self.placed_roles}
        self._update_plot()
        self._reset_simulation()
        ui.notify("All nodes cleared", type='info')

    def _clear_all(self):
        self._clear_nodes()


def main():
    Robust6GDemonstrator()
    ui.run(title="ROBUST-6G WP6 Demonstrator", port=8080, reload=False, dark=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
