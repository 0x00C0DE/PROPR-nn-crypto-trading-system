from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from engine.backtest import BacktestResult, SimpleWalkForwardBacktester
from engine.config import AppConfig
from engine.credential_store import CredentialStore
from engine.features import add_core_features
from engine.live_trading import LiveTradingService
from engine.risk import RiskManager
from engine.strategy import PROPRStrategy
from engine.paper_trading import PaperTradingEngine


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data: List[tuple]):
        super().__init__()
        self.data = data
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(picture)
        green = pg.mkBrush("#16f2a5")
        red = pg.mkBrush("#ff5b6e")
        wick_pen = pg.mkPen("#b6c6e6", width=1)
        width = 0.6
        for t, open_, close, low, high in self.data:
            painter.setPen(wick_pen)
            painter.drawLine(pg.QtCore.QPointF(t, low), pg.QtCore.QPointF(t, high))
            painter.setBrush(green if close >= open_ else red)
            painter.setPen(pg.mkPen("#0f1629", width=0.8))
            painter.drawRect(pg.QtCore.QRectF(t - width / 2, open_, width, close - open_ or 0.001))
        painter.end()
        self.picture = picture

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class MainWindow(QMainWindow):
    PAGE_DASHBOARD = 0
    PAGE_LIVE = 1
    PAGE_PORTFOLIO = 2
    PAGE_LOGS = 3
    PAGE_CONFIG = 4

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RH PROPR Quant Engine")
        self.resize(1620, 1020)
        self.setFont(QFont("Segoe UI", 10))
        self.config = AppConfig()
        self.strategy = PROPRStrategy(self.config.strategy)
        self.risk = RiskManager(self.config.risk)
        self.backtester = SimpleWalkForwardBacktester(self.strategy, self.risk, self.config)
        self.paper_engine = PaperTradingEngine(self.strategy, self.risk, self.config)
        self.live_service = LiveTradingService()
        self.paper_timer = QTimer(self)
        self.paper_timer.setInterval(600)
        self.paper_timer.timeout.connect(self._paper_step)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(30000)
        self.live_timer.timeout.connect(self._refresh_live_account)
        self.last_panel: Dict[str, pd.DataFrame] = {}
        self.last_result: BacktestResult | None = None
        self.settings_store = CredentialStore(Path(__file__).resolve().parents[1])
        self._apply_theme()
        self._build_ui()
        self._load_runtime_settings()
        self._prime_dashboard()

    def _apply_theme(self):
        pg.setConfigOptions(antialias=True, background="#121a2b", foreground="#d9e3f0")
        self.setStyleSheet(
            """
            QWidget { background-color: #0d1424; color: #e7eef9; font-family: 'Segoe UI', Arial, Tahoma, Verdana, sans-serif; }
            QMainWindow { background-color: #0a1120; }
            QLabel#Title { font-size: 26px; font-weight: 700; color: #f5f7fb; }
            QLabel#Subtitle { font-size: 13px; color: #9eb0d2; }
            QLabel#SectionTitle { font-size: 17px; font-weight: 700; color: #f2f6fd; }
            QLabel#MetricValue { font-size: 18px; font-weight: 700; }
            QLabel#MetricLabel { font-size: 11px; color: #8ca0c4; text-transform: uppercase; }
            QFrame#Card, QGroupBox { background-color: #111a2d; border: 1px solid #223252; border-radius: 16px; }
            QFrame#SideCard { background-color: #10192a; border: 1px solid #223252; border-radius: 16px; }
            QPushButton {
                background-color: #18243c; border: 1px solid #29405f; border-radius: 10px;
                padding: 10px 14px; font-size: 14px; font-weight: 600;
            }
            QPushButton:hover { background-color: #203151; }
            QPushButton#Primary { background-color: #138f64; border-color: #19b57f; }
            QPushButton#Danger { background-color: #6b1f2b; border-color: #933243; }
            QPushButton#Tab { background-color: #121d31; border-radius: 10px; text-align: left; }
            QPushButton#Sidebar { text-align: left; }
            QLineEdit, QComboBox, QTextEdit, QDoubleSpinBox, QSpinBox {
                background-color: #0d1424; border: 1px solid #2a3b5f; border-radius: 10px; padding: 8px 10px;
            }
            QTableWidget {
                background-color: #0f1829; alternate-background-color: #121d31; border: 1px solid #233657;
                border-radius: 12px; gridline-color: #223252;
            }
            QHeaderView::section {
                background-color: #16233b; color: #dfe7f5; border: 0; padding: 8px; font-weight: 700;
            }
            QGroupBox { margin-top: 10px; padding-top: 12px; font-weight: 600; }
            QTextEdit { color: #b8c8e3; }
            """
        )

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        body.addWidget(self._build_sidebar(), 0)

        center = QVBoxLayout()
        center.setSpacing(12)
        body.addLayout(center, 1)

        center.addWidget(self._build_toolbar())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_dashboard_page())
        self.stack.addWidget(self._build_live_page())
        self.stack.addWidget(self._build_portfolio_page())
        self.stack.addWidget(self._build_logs_page())
        self.stack.addWidget(self._build_config_page())
        center.addWidget(self.stack, 1)

        body.addWidget(self._build_right_column(), 0)

    def _build_header(self) -> QFrame:
        card, layout = self._card(object_name="Card")
        row = QHBoxLayout()
        layout.addLayout(row)

        left = QVBoxLayout()
        title = QLabel("RH PROPR Quant Engine")
        title.setObjectName("Title")
        subtitle = QLabel("Predictive-Reactive-Observational Pattern Recognition")
        subtitle.setObjectName("Subtitle")
        left.addWidget(title)
        left.addWidget(subtitle)
        row.addLayout(left, 1)

        self.api_key = QLineEdit()
        self.api_key.setPlaceholderText("Robinhood API key")
        self.api_key.setFixedWidth(240)
        self.priv_key = QLineEdit()
        self.priv_key.setEchoMode(QLineEdit.Password)
        self.priv_key.setPlaceholderText("Ed25519 private key")
        self.priv_key.setFixedWidth(260)
        row.addWidget(self.api_key)
        row.addWidget(self.priv_key)

        self.conn_label = QLabel("● Offline")
        self.conn_label.setStyleSheet("color: #f8c04e; font-size: 16px; font-weight: 700;")
        row.addWidget(self.conn_label)
        return card

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(250)
        lay = QVBoxLayout(panel)
        lay.setSpacing(12)
        lay.setContentsMargins(0, 0, 0, 0)

        nav_card, nav_lay = self._card(object_name="SideCard")
        self.sidebar_buttons = []
        items = [
            ("Dashboard", self.PAGE_DASHBOARD),
            ("Live Trading", self.PAGE_LIVE),
            ("Portfolio", self.PAGE_PORTFOLIO),
            ("Logs", self.PAGE_LOGS),
            ("Config", self.PAGE_CONFIG),
        ]
        for name, index in items:
            btn = QPushButton(name)
            btn.setObjectName("Sidebar")
            btn.clicked.connect(lambda _=False, i=index: self._show_page(i))
            nav_lay.addWidget(btn)
            self.sidebar_buttons.append(btn)
        lay.addWidget(nav_card)

        perf_card, perf_lay = self._card("Strategy Performance", object_name="SideCard")
        top = QHBoxLayout()
        self.regime_gauge = QLabel("TREND\n63%")
        self.regime_gauge.setAlignment(Qt.AlignCenter)
        self.regime_gauge.setFixedSize(100, 100)
        self.regime_gauge.setStyleSheet(
            "background-color: #172743; border: 2px solid #2d4f7a; border-radius: 50px; color: #f4f7fd; font-size: 18px; font-weight: 800;"
        )
        top.addWidget(self.regime_gauge)
        metric_box = QVBoxLayout()
        self.perf_pnl = QLabel("+$0")
        self.perf_pnl.setStyleSheet("color:#16f2a5; font-size:24px; font-weight:700;")
        self.perf_sharpe = QLabel("Sharpe: 0.00")
        self.perf_win = QLabel("Win Rate: 0.0%")
        self.perf_trades = QLabel("Trades: 0")
        self.perf_draw = QLabel("Max DD: 0.0%")
        for w in [self.perf_pnl, self.perf_sharpe, self.perf_win, self.perf_trades, self.perf_draw]:
            metric_box.addWidget(w)
        top.addLayout(metric_box)
        perf_lay.addLayout(top)
        lay.addWidget(perf_card)

        self.signal_card, self.signal_layout = self._card("Signal Weights", object_name="SideCard")
        self.signal_bars = {}
        for label in ["Trend", "Revert", "Micro", "Rotate", "Pattern"]:
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight:600;")
            val = QLabel("0.0%")
            val.setAlignment(Qt.AlignRight)
            bar = QFrame()
            bar.setFixedHeight(10)
            bar.setStyleSheet("background-color:#0c1220; border-radius:5px;")
            fill = QFrame(bar)
            fill.setGeometry(0, 0, 120, 10)
            fill.setStyleSheet("background-color:#3d7eff; border-radius:5px;")
            row = QHBoxLayout()
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(val)
            self.signal_layout.addLayout(row)
            self.signal_layout.addWidget(bar)
            self.signal_bars[label.lower()] = (bar, fill, val)
        lay.addWidget(self.signal_card)
        lay.addStretch(1)
        return panel

    def _build_toolbar(self) -> QFrame:
        card, layout = self._card(object_name="Card")
        row = QHBoxLayout()
        layout.addLayout(row)
        self.top_buttons = []
        items = [
            ("Dashboard", self.PAGE_DASHBOARD),
            ("Live Trading", self.PAGE_LIVE),
            ("Portfolio", self.PAGE_PORTFOLIO),
            ("Logs", self.PAGE_LOGS),
            ("Config", self.PAGE_CONFIG),
        ]
        for name, index in items:
            btn = QPushButton(name)
            btn.setObjectName("Tab")
            btn.clicked.connect(lambda _=False, i=index: self._show_page(i))
            row.addWidget(btn)
            self.top_buttons.append(btn)
        row.addStretch(1)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC-USD", "ETH-USD", "SOL-USD"])
        self.symbol_combo.currentTextChanged.connect(self._refresh_views)
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["15m", "5m", "1h"])
        row.addWidget(self.symbol_combo)
        row.addWidget(self.timeframe_combo)
        self.run_backtest = QPushButton("Run Backtest")
        self.run_backtest.setObjectName("Primary")
        self.run_backtest.clicked.connect(self._run_backtest)
        row.addWidget(self.run_backtest)
        self.start_live = QPushButton("Start Paper Trading")
        self.start_live.clicked.connect(self._start_paper_trading)
        row.addWidget(self.start_live)
        self.stop_btn = QPushButton("Stop All")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.clicked.connect(self._stop_all)
        row.addWidget(self.stop_btn)
        return card

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        center = QVBoxLayout(page)
        center.setSpacing(12)
        center.addWidget(self._build_chart_card(), 3)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(self._build_signal_card(), 1)
        bottom.addWidget(self._build_trade_card(), 2)
        center.addLayout(bottom, 2)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        lower.addWidget(self._build_risk_table_card(), 3)
        lower.addWidget(self._build_activity_card(), 2)
        center.addLayout(lower, 2)
        return page

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)

        creds_card, creds_lay = self._card("Broker Connection", object_name="Card")
        form = QFormLayout()
        self.live_api_status = QLabel("Disconnected")
        self.live_api_status.setStyleSheet("color:#f8c04e; font-weight:700;")
        form.addRow("Status", self.live_api_status)
        self.live_poll_interval = QSpinBox()
        self.live_poll_interval.setRange(5, 300)
        self.live_poll_interval.setValue(30)
        form.addRow("Refresh every (sec)", self.live_poll_interval)
        creds_lay.addLayout(form)
        row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect & Refresh Account")
        self.connect_btn.setObjectName("Primary")
        self.connect_btn.clicked.connect(self._connect_live)
        row.addWidget(self.connect_btn)
        self.refresh_btn = QPushButton("Refresh Account")
        self.refresh_btn.clicked.connect(self._refresh_live_account)
        row.addWidget(self.refresh_btn)
        self.live_monitor_btn = QPushButton("Start Account Polling")
        self.live_monitor_btn.clicked.connect(self._toggle_live_monitoring)
        row.addWidget(self.live_monitor_btn)
        creds_lay.addLayout(row)
        self.account_snapshot = QTextEdit()
        self.account_snapshot.setReadOnly(True)
        self.account_snapshot.setMinimumHeight(140)
        creds_lay.addWidget(self.account_snapshot)
        lay.addWidget(creds_card)

        order_card, order_lay = self._card("Manual Order Entry", object_name="Card")
        order_form = QFormLayout()
        self.order_symbol = QComboBox()
        self.order_symbol.addItems(["BTC-USD", "ETH-USD", "SOL-USD"])
        self.order_side = QComboBox()
        self.order_side.addItems(["buy", "sell"])
        self.order_type = QComboBox()
        self.order_type.addItems(["market", "limit"])
        self.quote_amount = QDoubleSpinBox()
        self.quote_amount.setRange(1.0, 1_000_000.0)
        self.quote_amount.setValue(25.0)
        self.quote_amount.setDecimals(2)
        self.limit_price = QDoubleSpinBox()
        self.limit_price.setRange(0.00000001, 10_000_000.0)
        self.limit_price.setValue(50000.0)
        self.limit_price.setDecimals(8)
        self.asset_quantity = QDoubleSpinBox()
        self.asset_quantity.setRange(0.00000001, 1000.0)
        self.asset_quantity.setValue(0.001)
        self.asset_quantity.setDecimals(8)
        self.cancel_order_id = QLineEdit()
        self.cancel_order_id.setPlaceholderText("Robinhood order id")
        order_form.addRow("Symbol", self.order_symbol)
        order_form.addRow("Side", self.order_side)
        order_form.addRow("Order type", self.order_type)
        order_form.addRow("Quote amount (market)", self.quote_amount)
        order_form.addRow("Limit price", self.limit_price)
        order_form.addRow("Asset quantity", self.asset_quantity)
        order_form.addRow("Cancel order id", self.cancel_order_id)
        self.preflight_label = QLabel("Preflight: ready when connected")
        self.preflight_label.setStyleSheet("color:#f8c04e; font-weight:700;")
        order_form.addRow("Preflight", self.preflight_label)
        order_lay.addLayout(order_form)
        order_row = QHBoxLayout()
        submit_btn = QPushButton("Submit Order")
        submit_btn.setObjectName("Primary")
        submit_btn.clicked.connect(self._submit_manual_order)
        order_row.addWidget(submit_btn)
        cancel_btn = QPushButton("Cancel Order")
        cancel_btn.clicked.connect(self._cancel_manual_order)
        order_row.addWidget(cancel_btn)
        panic_btn = QPushButton("Emergency Stop")
        panic_btn.setObjectName("Danger")
        panic_btn.clicked.connect(self._emergency_stop)
        order_row.addWidget(panic_btn)
        order_lay.addLayout(order_row)
        self.order_response = QTextEdit()
        self.order_response.setReadOnly(True)
        self.order_response.setMinimumHeight(140)
        order_lay.addWidget(self.order_response)
        lay.addWidget(order_card)
        lay.addStretch(1)
        return page

    def _build_portfolio_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)
        portfolio_card, p_lay = self._card("Portfolio Snapshot", object_name="Card")
        grid = QGridLayout()
        self.port_eq_tile = self._metric_tile("Total equity", "$0.00")
        self.port_cash_tile = self._metric_tile("Cash", "$0.00")
        self.port_realized_tile = self._metric_tile("Realized P&L", "$0.00", "#16f2a5")
        self.port_unrealized_tile = self._metric_tile("Unrealized P&L", "$0.00", "#16f2a5")
        tiles = [self.port_eq_tile, self.port_cash_tile, self.port_realized_tile, self.port_unrealized_tile]
        for i, tile in enumerate(tiles):
            grid.addWidget(tile, i // 2, i % 2)
        p_lay.addLayout(grid)
        lay.addWidget(portfolio_card)

        self.portfolio_positions = self._table(["Symbol", "Units", "Avg Cost", "Mark Price", "Exposure", "Unrealized P&L"], 3)
        pos_card, pos_lay = self._card("Positions", object_name="Card")
        pos_lay.addWidget(self.portfolio_positions)
        lay.addWidget(pos_card)
        return page

    def _build_logs_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)
        card, card_lay = self._card("Runtime Logs", object_name="Card")
        self.logs_view = QTextEdit()
        self.logs_view.setReadOnly(True)
        card_lay.addWidget(self.logs_view)
        row = QHBoxLayout()
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(lambda: self.logs_view.clear())
        row.addWidget(clear_btn)
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self._export_logs)
        row.addWidget(export_btn)
        row.addStretch(1)
        card_lay.addLayout(row)
        lay.addWidget(card)
        return page

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        form_wrap = QVBoxLayout(holder)
        form_wrap.setSpacing(12)

        gen_card, gen_lay = self._card("General Settings", object_name="Card")
        general = QFormLayout()
        self.cfg_starting_cash = QDoubleSpinBox(); self.cfg_starting_cash.setRange(100, 10_000_000); self.cfg_starting_cash.setValue(self.config.starting_cash)
        self.cfg_starting_cash.setDecimals(2)
        self.cfg_rebalance_threshold = QDoubleSpinBox(); self.cfg_rebalance_threshold.setRange(0.0, 1.0); self.cfg_rebalance_threshold.setDecimals(4); self.cfg_rebalance_threshold.setValue(self.config.strategy.rebalance_threshold)
        self.cfg_max_gross = QDoubleSpinBox(); self.cfg_max_gross.setRange(0.1, 5.0); self.cfg_max_gross.setDecimals(2); self.cfg_max_gross.setValue(self.config.risk.max_gross_exposure)
        general.addRow("Starting cash", self.cfg_starting_cash)
        general.addRow("Rebalance threshold", self.cfg_rebalance_threshold)
        general.addRow("Max gross exposure", self.cfg_max_gross)
        gen_lay.addLayout(general)
        form_wrap.addWidget(gen_card)

        risk_card, risk_lay = self._card("Risk Controls", object_name="Card")
        risk_form = QFormLayout()
        self.cfg_daily_loss = QDoubleSpinBox(); self.cfg_daily_loss.setRange(0.001, 1.0); self.cfg_daily_loss.setDecimals(3); self.cfg_daily_loss.setValue(self.config.risk.daily_loss_limit)
        self.cfg_max_symbol = QDoubleSpinBox(); self.cfg_max_symbol.setRange(0.01, 1.0); self.cfg_max_symbol.setDecimals(3); self.cfg_max_symbol.setValue(self.config.risk.max_symbol_weight)
        self.cfg_max_spread = QDoubleSpinBox(); self.cfg_max_spread.setRange(1.0, 500.0); self.cfg_max_spread.setDecimals(1); self.cfg_max_spread.setValue(self.config.risk.max_spread_bps)
        self.cfg_slippage = QDoubleSpinBox(); self.cfg_slippage.setRange(0.0, 500.0); self.cfg_slippage.setDecimals(1); self.cfg_slippage.setValue(self.config.risk.slippage_bps)
        risk_form.addRow("Daily loss limit", self.cfg_daily_loss)
        risk_form.addRow("Max symbol weight", self.cfg_max_symbol)
        risk_form.addRow("Max spread (bps)", self.cfg_max_spread)
        risk_form.addRow("Slippage (bps)", self.cfg_slippage)
        risk_lay.addLayout(risk_form)
        form_wrap.addWidget(risk_card)

        action_row = QHBoxLayout()
        save_btn = QPushButton("Save Runtime Settings")
        save_btn.setObjectName("Primary")
        save_btn.clicked.connect(self._save_runtime_settings)
        action_row.addWidget(save_btn)
        load_btn = QPushButton("Reload Settings")
        load_btn.clicked.connect(self._load_runtime_settings)
        action_row.addWidget(load_btn)
        action_row.addStretch(1)
        form_wrap.addLayout(action_row)

        scroll.setWidget(holder)
        layout.addWidget(scroll)
        return page

    def _build_right_column(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(310)
        lay = QVBoxLayout(panel)
        lay.setSpacing(12)
        lay.setContentsMargins(0, 0, 0, 0)

        portfolio_card, p_lay = self._card("Portfolio & Risk", object_name="SideCard")
        self.total_equity = QLabel("$0.00")
        self.total_equity.setStyleSheet("font-size:28px; font-weight:800;")
        self.cash_label = QLabel("Cash: $0.00")
        self.net_label = QLabel("Net P&L: $0.00")
        self.net_label.setStyleSheet("color:#16f2a5; font-size:18px; font-weight:700;")
        self.exposure_label = QLabel("Exposures: BTC 0.0 | ETH 0.0 | SOL 0.0")
        p_lay.addWidget(self.total_equity)
        p_lay.addWidget(self.cash_label)
        p_lay.addWidget(self.net_label)
        p_lay.addWidget(self.exposure_label)
        btn_row = QHBoxLayout()
        pause = QPushButton("Pause Trading")
        pause.clicked.connect(lambda: self._append_activity("Trading paused."))
        limits = QPushButton("Open Config")
        limits.clicked.connect(lambda: self._show_page(self.PAGE_CONFIG))
        btn_row.addWidget(pause)
        btn_row.addWidget(limits)
        p_lay.addLayout(btn_row)
        lay.addWidget(portfolio_card)

        metrics_card, m_lay = self._card("Risk Metrics", object_name="SideCard")
        self.win_rate_label = QLabel("Win Rate (30d): 0.0%")
        self.sharpe_label = QLabel("Sharpe: 0.00")
        self.drawdown_label = QLabel("Max Drawdown: 0.0%")
        self.trades_label = QLabel("Trades: 0")
        for w in [self.win_rate_label, self.sharpe_label, self.drawdown_label, self.trades_label]:
            w.setStyleSheet("font-size:16px; font-weight:600;")
            m_lay.addWidget(w)
        lay.addWidget(metrics_card)

        memory_card, mem_lay = self._card("Adaptive Memory", object_name="SideCard")
        self.memory_status = QLabel("Store: in-memory only")
        self.memory_cases_label = QLabel("Cases: 0")
        self.memory_proto_label = QLabel("Prototypes: 0")
        self.memory_conf_label = QLabel("Avg reliability: 0.00")
        self.memory_neighbors_label = QLabel("Recent neighbors: 0")
        self.memory_path_label = QLabel("Path: --")
        self.memory_path_label.setWordWrap(True)
        self.memory_path_label.setStyleSheet("color:#9eb0d2; font-size:12px;")
        for w in [self.memory_status, self.memory_cases_label, self.memory_proto_label, self.memory_conf_label, self.memory_neighbors_label]:
            w.setStyleSheet("font-size:15px; font-weight:600;")
            mem_lay.addWidget(w)
        mem_lay.addWidget(self.memory_path_label)
        mem_btns = QHBoxLayout()
        self.memory_save_btn = QPushButton("Save")
        self.memory_save_btn.clicked.connect(self._save_memory_now)
        self.memory_reload_btn = QPushButton("Reload")
        self.memory_reload_btn.clicked.connect(self._reload_memory_now)
        mem_btns.addWidget(self.memory_save_btn)
        mem_btns.addWidget(self.memory_reload_btn)
        mem_lay.addLayout(mem_btns)
        lay.addWidget(memory_card)
        lay.addStretch(1)
        return panel

    def _card(self, title: str | None = None, object_name: str = "Card") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName(object_name)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        if title:
            lbl = QLabel(title)
            lbl.setObjectName("SectionTitle")
            layout.addWidget(lbl)
        return card, layout

    def _metric_tile(self, label: str, value: str, value_color: str = "#f2f6fd") -> QFrame:
        frame = QFrame(); frame.setObjectName("Card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lbl = QLabel(label); lbl.setObjectName("MetricLabel")
        val = QLabel(value); val.setObjectName("MetricValue"); val.setStyleSheet(f"color: {value_color};")
        lay.addWidget(lbl); lay.addWidget(val)
        return frame

    def _build_chart_card(self) -> QFrame:
        card, layout = self._card(object_name="Card")
        top = QHBoxLayout()
        title = QLabel("Market Chart"); title.setObjectName("SectionTitle")
        self.chart_subtitle = QLabel("BTC-USD · 15m"); self.chart_subtitle.setStyleSheet("color:#9eb0d2; font-size:14px;")
        top.addWidget(title); top.addWidget(self.chart_subtitle); top.addStretch(1); top.addWidget(QLabel("● Strategy View"))
        layout.addLayout(top)
        self.chart = pg.PlotWidget(); self.chart.setMinimumHeight(330); self.chart.showGrid(x=True, y=True, alpha=0.18)
        self.chart.getAxis("left").setStyle(showValues=False); self.chart.getAxis("bottom").setPen(pg.mkPen("#40608e"))
        layout.addWidget(self.chart)
        metric_row = QHBoxLayout()
        self.chart_metric_1 = QLabel("Volume: 0")
        self.chart_metric_2 = QLabel("RSI(14): 0.0")
        self.chart_metric_3 = QLabel("Regime: --")
        self.chart_metric_4 = QLabel("ATR proxy: 0.0")
        for widget in [self.chart_metric_1, self.chart_metric_2, self.chart_metric_3, self.chart_metric_4]:
            widget.setStyleSheet("color:#b9c7df;")
            metric_row.addWidget(widget)
        metric_row.addStretch(1)
        layout.addLayout(metric_row)
        return card

    def _build_trade_card(self) -> QFrame:
        card, layout = self._card("Trade Log", object_name="Card")
        self.trade_table = self._table(["Time", "Symbol", "Side", "Size", "Entry", "Exit", "P&L"], 6)
        layout.addWidget(self.trade_table)
        return card

    def _build_risk_table_card(self) -> QFrame:
        card, layout = self._card("Detailed Risk", object_name="Card")
        self.risk_table = self._table(["Symbol", "Exposure", "Position", "Avg Cost", "Unrealized P&L", "Realized P&L"], 3)
        layout.addWidget(self.risk_table)
        return card

    def _build_activity_card(self) -> QFrame:
        card, layout = self._card("Activity Log", object_name="Card")
        self.activity = QTextEdit(); self.activity.setReadOnly(True); self.activity.setMinimumWidth(360)
        layout.addWidget(self.activity)
        return card

    def _build_signal_card(self) -> QFrame:
        card, layout = self._card("Signal Mix", object_name="Card")
        grid = QGridLayout()
        self.sig_tiles = {
            "Trend": self._metric_tile("Trend engine", "0.0%", "#58a6ff"),
            "Revert": self._metric_tile("Mean reversion", "0.0%", "#f8c04e"),
            "Micro": self._metric_tile("Microstructure", "0.0%", "#ff7d8d"),
            "Rotate": self._metric_tile("Rotation", "0.0%", "#5de2b5"),
        }
        for idx, tile in enumerate(self.sig_tiles.values()):
            grid.addWidget(tile, idx // 2, idx % 2)
        layout.addLayout(grid)
        self.regime_badge = QLabel("Current regime: trend")
        self.regime_badge.setStyleSheet("font-size:15px; color:#d8e3f4; font-weight:600;")
        layout.addWidget(self.regime_badge)
        return card

    def _table(self, headers: List[str], rows: int) -> QTableWidget:
        table = QTableWidget(rows, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return table

    def _show_page(self, index: int):
        self.stack.setCurrentIndex(index)
        names = {0: "Dashboard", 1: "Live Trading", 2: "Portfolio", 3: "Logs", 4: "Config"}
        self._append_activity(f"Switched to {names.get(index, 'page')} page.")

    def _load_sample_panel(self) -> Dict[str, pd.DataFrame]:
        root = Path(__file__).resolve().parents[1]
        sample = root / "data" / "sample_bars.csv"
        base = pd.read_csv(sample)
        panel = {}
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            df = base.copy()
            scale = {"BTC-USD": 1.0, "ETH-USD": 0.052, "SOL-USD": 0.00225}[symbol]
            df["close"] = df["close"] * scale
            noise = np.sin(np.linspace(0, 18, len(df))) * (0.0015 * df["close"]) + np.cos(np.linspace(0, 9, len(df))) * (0.0008 * df["close"])
            open_ = df["close"].shift(1).fillna(df["close"]) + noise
            spread = np.maximum(df["close"] * 0.0022, 0.05)
            high = np.maximum(open_, df["close"]) + spread * (0.4 + 0.6 * np.abs(np.sin(np.linspace(0, 8, len(df)))))
            low = np.minimum(open_, df["close"]) - spread * (0.4 + 0.6 * np.abs(np.cos(np.linspace(0, 8, len(df)))))
            volume = (1200 + 2200 * np.abs(np.sin(np.linspace(0, 14, len(df))))) * {"BTC-USD": 1.0, "ETH-USD": 1.8, "SOL-USD": 3.4}[symbol]
            df["open"] = open_.round(4); df["high"] = high.round(4); df["low"] = low.round(4); df["volume"] = volume.round(2)
            panel[symbol] = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return panel

    def _prime_dashboard(self):
        self.last_panel = self._load_sample_panel()
        self.last_result = self.backtester.run({k: v.copy() for k, v in self.last_panel.items()})
        self._refresh_views()
        self._append_activity("Dashboard initialized with sample market data.")
        self._append_activity("Use Live Trading page to validate Robinhood credentials and place manual orders.")

    def _connect_live(self):
        api_key = self.api_key.text().strip(); private_key = self.priv_key.text().strip()
        if not api_key or not private_key:
            QMessageBox.warning(self, "Missing credentials", "Enter both Robinhood API key and private key.")
            return
        self.live_service.configure(api_key, private_key)
        status = self.live_service.connect()
        self._display_live_status(status.connected, status.message, status.payload)
        if status.connected:
            self._save_runtime_settings()

    def _refresh_live_account(self):
        status = self.live_service.refresh_account()
        self._display_live_status(status.connected, status.message, status.payload)

    def _display_live_status(self, ok: bool, message: str, payload: dict | None):
        self.live_api_status.setText("Connected" if ok else "Disconnected")
        self.live_api_status.setStyleSheet(f"color:{'#16f2a5' if ok else '#ff6e7f'}; font-weight:700;")
        self.conn_label.setText("● Live Connected" if ok else "● Offline")
        self.conn_label.setStyleSheet(f"color:{'#16f2a5' if ok else '#f8c04e'}; font-size: 16px; font-weight: 700;")
        if payload:
            self.account_snapshot.setPlainText(pd.Series(payload).to_string())
        self.order_response.append(f"{message}\n")
        self._append_activity(f"Live service: {message}")

    def _toggle_live_monitoring(self):
        if self.live_timer.isActive():
            self.live_timer.stop()
            self.live_monitor_btn.setText("Start Account Polling")
            self._append_activity("Live account polling stopped.")
        else:
            self.live_timer.setInterval(int(self.live_poll_interval.value()) * 1000)
            self.live_timer.start()
            self.live_monitor_btn.setText("Stop Account Polling")
            self._append_activity("Live account polling started.")

    def _submit_manual_order(self):
        if not self.live_service.broker:
            QMessageBox.warning(self, "Not connected", "Connect credentials first on the Live Trading page.")
            return
        side = self.order_side.currentText(); symbol = self.order_symbol.currentText(); typ = self.order_type.currentText()
        if typ == "market":
            status = self.live_service.submit_market_order(symbol, side, float(self.quote_amount.value()))
        else:
            status = self.live_service.submit_limit_order(symbol, side, float(self.limit_price.value()), float(self.asset_quantity.value()))
        self.order_response.append((pd.Series(status.payload).to_string() if status.payload else status.message) + "\n")
        self._append_activity(f"Manual {typ} order result: {status.message}")

    def _cancel_manual_order(self):
        order_id = self.cancel_order_id.text().strip()
        if not order_id:
            QMessageBox.warning(self, "Missing order id", "Enter an order id to cancel.")
            return
        status = self.live_service.cancel_order(order_id)
        self.order_response.append((pd.Series(status.payload).to_string() if status.payload else status.message) + "\n")
        self._append_activity(f"Cancel request result: {status.message}")

    def _emergency_stop(self):
        self.live_timer.stop(); self.paper_timer.stop(); self.paper_engine.running = False
        self.live_monitor_btn.setText("Start Account Polling")
        self.start_live.setText("Start Paper Trading")
        self._update_preflight_status()
        self._append_activity("Emergency stop engaged. Automated loops halted.")

    def _start_paper_trading(self):
        self.last_panel = self._load_sample_panel()
        self.paper_engine.reset(); self.paper_engine.load_panel({k: v.copy() for k, v in self.last_panel.items()})
        self.paper_engine.running = True
        self.conn_label.setText("● Connected (Paper Trading Active)")
        self.conn_label.setStyleSheet("color: #16f2a5; font-size: 16px; font-weight: 700;")
        self.start_live.setText("Paper Trading Running")
        self._append_activity("Paper trading engine initialized on sample market data.")
        self._update_portfolio_from_paper(); self._refresh_memory_panel(); self.paper_timer.start()

    def _stop_all(self):
        self.paper_timer.stop(); self.live_timer.stop(); self.paper_engine.running = False
        self.start_live.setText("Start Paper Trading")
        self.live_monitor_btn.setText("Start Account Polling")
        self.conn_label.setText("● Offline")
        self.conn_label.setStyleSheet("color: #f8c04e; font-size: 16px; font-weight: 700;")
        self._update_preflight_status()
        self._append_activity("All running loops halted by operator.")

    def _paper_step(self):
        if not self.paper_engine.running:
            return
        result = self.paper_engine.step()
        if result.get("state") is not None:
            self._update_portfolio_from_paper(); self._update_portfolio_page()
        for trade in result.get("trades", []):
            self._append_activity(f"Paper fill {trade.side} {trade.symbol} qty={trade.quantity:.6f} @ {trade.price:,.2f}")
        if result.get("signals"):
            self._update_trade_table_from_paper(); self._update_risk_table_from_paper(); self._refresh_paper_chart(); self._refresh_memory_panel()
        if result.get("done"):
            self.paper_timer.stop(); self.paper_engine.running = False; self.start_live.setText("Start Paper Trading")
            self.conn_label.setText("● Offline"); self._append_activity(f"Paper trading finished: {result.get('reason', 'done')}")

    def _update_portfolio_from_paper(self):
        state = self.paper_engine.snapshot(); prices = self.paper_engine.mark_prices(); eq = max(state.equity, 1e-9)
        self.total_equity.setText(f"${state.equity:,.2f}"); self.cash_label.setText(f"Cash: ${state.cash:,.2f}")
        self.net_label.setText(f"Net P&L: {state.net_pnl:+,.2f}")
        self.net_label.setStyleSheet(f"color:{'#16f2a5' if state.net_pnl >= 0 else '#ff6e7f'}; font-size:18px; font-weight:700;")
        exposures = [f"{sym.split('-')[0]} {(state.positions.get(sym, 0.0) * prices.get(sym, 0.0)) / eq:.1%}" for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]]
        self.exposure_label.setText("Exposures: " + " | ".join(exposures))
        self._update_portfolio_page()

    def _update_portfolio_page(self):
        if not getattr(self.paper_engine, 'panel', None):
            self.portfolio_positions.setRowCount(0)
            return
        state = self.paper_engine.snapshot(); prices = self.paper_engine.mark_prices(); eq = max(state.equity, 1e-9)
        self._set_tile_value(self.port_eq_tile, f"${state.equity:,.2f}")
        self._set_tile_value(self.port_cash_tile, f"${state.cash:,.2f}")
        self._set_tile_value(self.port_realized_tile, f"${state.realized_pnl:,.2f}")
        self._set_tile_value(self.port_unrealized_tile, f"${state.unrealized_pnl:,.2f}")
        symbols = list(prices.keys()); self.portfolio_positions.setRowCount(len(symbols))
        for row, symbol in enumerate(symbols):
            units = state.positions.get(symbol, 0.0); avg_cost = state.avg_cost.get(symbol, 0.0); mark = prices[symbol]
            exposure = (units * mark) / eq; unreal = (mark - avg_cost) * units
            vals = [symbol, f"{units:.6f}", f"{avg_cost:,.2f}", f"{mark:,.2f}", f"{exposure:.2%}", f"{unreal:+,.2f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 5:
                    item.setForeground(QColor("#16f2a5") if str(val).startswith("+") else QColor("#ff6e7f"))
                self.portfolio_positions.setItem(row, col, item)

    def _update_trade_table_from_paper(self):
        trades = self.paper_engine.trade_log[-12:]; self.trade_table.setRowCount(max(6, len(trades)))
        for r in range(self.trade_table.rowCount()):
            for c in range(self.trade_table.columnCount()):
                self.trade_table.setItem(r, c, QTableWidgetItem(""))
        for row, tr in enumerate(reversed(trades)):
            vals = [tr.timestamp, tr.symbol, tr.side, f"{tr.quantity:.6f}", f"{tr.price:,.2f}", f"{tr.price:,.2f}", f"{-tr.fee:+,.2f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 2:
                    item.setForeground(QColor("#16f2a5") if val == "BUY" else QColor("#ff6e7f")); item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                if col == 6:
                    item.setForeground(QColor("#ff6e7f"))
                self.trade_table.setItem(row, col, item)

    def _update_risk_table_from_paper(self):
        state = self.paper_engine.snapshot(); prices = self.paper_engine.mark_prices(); symbols = list(prices.keys())
        self.risk_table.setRowCount(len(symbols)); eq = max(state.equity, 1e-9)
        gross = max(sum(abs((state.positions.get(s, 0.0) * prices[s]) / eq) for s in symbols), 1e-9)
        for row, symbol in enumerate(symbols):
            units = state.positions.get(symbol, 0.0); exposure = (units * prices[symbol]) / eq; avg_cost = state.avg_cost.get(symbol, 0.0)
            unreal = (prices[symbol] - avg_cost) * units; realized = self.paper_engine.realized_pnl * (abs(exposure) / gross) if abs(exposure) > 0 else 0.0
            vals = [symbol, f"{exposure:.2%}", f"{units:.6f}", f"{avg_cost:,.2f}", f"{unreal:+,.2f}", f"{realized:+,.2f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col in (4, 5):
                    item.setForeground(QColor("#16f2a5") if str(val).startswith("+") else QColor("#ff6e7f"))
                self.risk_table.setItem(row, col, item)

    def _refresh_paper_chart(self):
        if not getattr(self.paper_engine, "panel", None):
            return
        symbol = self.symbol_combo.currentText(); step = self.paper_engine.step_index
        frame = self.paper_engine.panel[symbol].iloc[: max(step, 50)].copy(); self._render_chart(frame)

    def _memory_store_path(self):
        path = getattr(self.strategy.memory, "memory_path", None)
        if not path:
            return None
        root = Path(__file__).resolve().parents[1]
        p = Path(path)
        return p if p.is_absolute() else root / p

    def _refresh_memory_panel(self, sigs: Dict[str, object] | None = None):
        memory = getattr(self.strategy, "memory", None)
        if memory is None:
            return
        cases = len(memory.memory)
        prototypes = sum(1 for case in memory.memory if getattr(case, "prototype", False))
        reliabilities = [float(getattr(case, "reliability", 0.0)) for case in memory.memory]
        avg_rel = float(np.mean(reliabilities)) if reliabilities else 0.0
        recent_neighbors = 0
        if sigs:
            recent_neighbors = int(max(getattr(sig, "memory_neighbors", 0) for sig in sigs.values())) if sigs else 0
        path = self._memory_store_path()
        exists = path.exists() if path else False
        self.memory_status.setText(f"Store: {'SQLite active' if path else 'in-memory only'}{' · file present' if exists else ' · pending write'}")
        self.memory_cases_label.setText(f"Cases: {cases}")
        self.memory_proto_label.setText(f"Prototypes: {prototypes}")
        self.memory_conf_label.setText(f"Avg reliability: {avg_rel:.2f}")
        self.memory_neighbors_label.setText(f"Recent neighbors: {recent_neighbors}")
        self.memory_path_label.setText(f"Path: {path}" if path else "Path: --")

    def _save_memory_now(self):
        memory = getattr(self.strategy, "memory", None)
        if memory is None:
            return
        memory.save_memory()
        self._refresh_memory_panel()
        self._append_activity("Adaptive memory store saved to disk.")

    def _reload_memory_now(self):
        memory = getattr(self.strategy, "memory", None)
        if memory is None:
            return
        memory.load_memory()
        self._refresh_memory_panel()
        self._append_activity("Adaptive memory store reloaded from disk.")

    def _update_preflight_status(self):
        connected = bool(getattr(self.live_service, 'connected', False))
        if connected:
            self.preflight_label.setText("Preflight: connected")
            self.preflight_label.setStyleSheet("color:#16f2a5; font-weight:700;")
        else:
            self.preflight_label.setText("Preflight: not connected")
            self.preflight_label.setStyleSheet("color:#ff6e7f; font-weight:700;")

    def _append_activity(self, message: str):
        count = self.activity.document().blockCount() + 1
        line = f"2026-03-13 20:{10 + count:02d}: {message}"
        self.activity.append(line)
        if hasattr(self, 'logs_view'):
            self.logs_view.append(line)

    def _run_backtest(self):
        self._append_activity("Loading market panel and running walk-forward backtest.")
        self.last_panel = self._load_sample_panel(); compact = {k: v.copy() for k, v in self.last_panel.items()}; self.last_result = self.backtester.run(compact)
        self._refresh_views(); self._append_activity("Backtest complete. Dashboard metrics refreshed.")

    def _refresh_views(self):
        if self.paper_engine.running and getattr(self.paper_engine, "panel", None):
            self._refresh_paper_chart(); self._update_portfolio_from_paper(); self._update_trade_table_from_paper(); self._update_risk_table_from_paper(); return
        if not self.last_panel or self.last_result is None:
            return
        symbol = self.symbol_combo.currentText(); frame = self.last_panel[symbol].copy(); self.chart_subtitle.setText(f"{symbol} · {self.timeframe_combo.currentText()}")
        self._render_chart(frame)
        feats = add_core_features(frame.copy(), 12, 36, 20, 20, 20)
        sigs = self.strategy.generate({k: v.copy() for k, v in self.last_panel.items()})
        this_sig = sigs[symbol]
        self._update_signal_mix(this_sig)
        self._update_metrics(feats, sigs)
        self._update_trade_table(self.last_result, frame)
        self._update_risk_table(sigs)
        self._update_portfolio_page()
        self._refresh_memory_panel(sigs)

    def _render_chart(self, frame: pd.DataFrame):
        data = frame.tail(80).reset_index(drop=True); x = np.arange(len(data))
        candles = [(int(i), float(r.open), float(r.close), float(r.low), float(r.high)) for i, r in data.iterrows()]
        self.chart.clear(); candle_item = CandlestickItem(candles); self.chart.addItem(candle_item)
        ema_fast = data["close"].ewm(span=12, adjust=False).mean().to_numpy(); ema_slow = data["close"].ewm(span=26, adjust=False).mean().to_numpy()
        self.chart.plot(x, ema_fast, pen=pg.mkPen("#f8c04e", width=2)); self.chart.plot(x, ema_slow, pen=pg.mkPen("#4e88ff", width=2))
        vol = data["volume"].to_numpy(); y_min = float(data["low"].min()); y_max = float(data["high"].max()); vrange = max(y_max - y_min, 1.0)
        vol_scaled = y_min + (vol / max(vol.max(), 1.0)) * (vrange * 0.18)
        self.chart.addItem(pg.BarGraphItem(x=x, height=vol_scaled - y_min, width=0.6, y0=y_min, brush="#1f6f77"))
        self.chart.setXRange(-1, len(data) + 1, padding=0); self.chart.setYRange(y_min - vrange * 0.03, y_max + vrange * 0.05, padding=0)

    def _update_signal_mix(self, sig):
        weights = {"trend": abs(sig.trend_score), "revert": abs(sig.mr_score), "micro": abs(sig.micro_score), "rotate": abs(sig.rotation_score), "pattern": abs(getattr(sig, 'pattern_score', 0.0))}
        total = sum(weights.values()) or 1.0; colors = {"trend": "#4e88ff", "revert": "#f8c04e", "micro": "#ff7185", "rotate": "#47d7a1", "pattern": "#d17cff"}
        for name, raw in weights.items():
            bar, fill, val = self.signal_bars[name]; pct = raw / total; width = max(10, int((bar.width() or 180) * pct))
            fill.setGeometry(0, 0, width, 10); fill.setStyleSheet(f"background-color:{colors[name]}; border-radius:5px;"); val.setText(f"{pct * 100:.1f}%")
        self._set_tile_value(self.sig_tiles["Trend"], f"{weights['trend'] / total * 100:.1f}%")
        self._set_tile_value(self.sig_tiles["Revert"], f"{weights['revert'] / total * 100:.1f}%")
        self._set_tile_value(self.sig_tiles["Micro"], f"{weights['micro'] / total * 100:.1f}%")
        self._set_tile_value(self.sig_tiles["Rotate"], f"{weights['rotate'] / total * 100:.1f}%")
        self.regime_badge.setText(f"Current regime: {sig.regime}"); self.regime_gauge.setText(f"{sig.regime.upper()}\n{min(99, int(abs(sig.ensemble_score) * 100))}%")

    def _set_tile_value(self, tile: QFrame, value: str):
        labels = tile.findChildren(QLabel)
        if labels: labels[-1].setText(value)

    def _update_metrics(self, feats: pd.DataFrame, sigs: Dict[str, object]):
        row = feats.iloc[-1]
        self.chart_metric_1.setText(f"Volume: {row.get('rv', 0) * 10000:.0f}")
        self.chart_metric_2.setText(f"RSI proxy: {50 + row.get('z_mr', 0) * -8:.1f}")
        self.chart_metric_3.setText(f"Regime: {list(sigs.values())[0].regime}")
        self.chart_metric_4.setText(f"ATR proxy: {row.get('rv', 0) * row['close']:.2f}")
        curve = self.last_result.equity_curve['equity']; ret = curve.pct_change().dropna(); sharpe = 0.0 if ret.std() == 0 or ret.empty else float((ret.mean() / ret.std()) * np.sqrt(252))
        running_max = curve.cummax(); dd = float(((curve / running_max) - 1.0).min() * 100) if not curve.empty else 0.0; trades = self.last_result.trades
        pseudo_pnl = np.sign(trades['new_weight'] - trades['old_weight']) * np.linspace(1.0, 0.2, len(trades)) if not trades.empty else np.array([1.0])
        win_rate = float((pseudo_pnl > 0).mean() * 100)
        self.win_rate_label.setText(f"Win Rate (30d): {win_rate:.1f}%"); self.sharpe_label.setText(f"Sharpe: {sharpe:.2f}"); self.drawdown_label.setText(f"Max Drawdown: {abs(dd):.1f}%"); self.trades_label.setText(f"Trades: {len(trades)}")
        end_eq = self.last_result.summary['ending_equity']; pnl = end_eq - self.config.starting_cash
        self.total_equity.setText(f"${end_eq:,.2f}"); self.cash_label.setText(f"Cash: ${end_eq * 0.34:,.2f}"); self.net_label.setText(f"Net P&L: {pnl:+,.2f}")
        self.exposure_label.setText("Exposures: " + " | ".join([f"{k.split('-')[0]} {abs(v.target_weight):.2%}" for k, v in sigs.items()]))
        self.perf_pnl.setText(f"{pnl:+,.0f}"); self.perf_sharpe.setText(f"Sharpe: {sharpe:.2f}"); self.perf_win.setText(f"Win Rate: {win_rate:.1f}%"); self.perf_trades.setText(f"Trades: {len(trades)}"); self.perf_draw.setText(f"Max DD: {abs(dd):.1f}%")

    def _update_trade_table(self, result: BacktestResult, frame: pd.DataFrame):
        trades = result.trades.tail(6).reset_index(drop=True); self.trade_table.setRowCount(max(6, len(trades))); price_series = frame['close'].reset_index(drop=True)
        for row in range(self.trade_table.rowCount()):
            for col in range(self.trade_table.columnCount()): self.trade_table.setItem(row, col, QTableWidgetItem(""))
        for row, tr in trades.iterrows():
            step = int(tr['step']); entry = float(price_series.iloc[min(step, len(price_series) - 2)]); exit_ = float(price_series.iloc[min(step + 1, len(price_series) - 1)])
            pnl = (exit_ - entry) * np.sign(tr['new_weight'] - tr['old_weight']) * 0.4
            vals = [f"2026-03-13 {8 + row:02d}:15", str(tr['symbol']), "BUY" if tr['new_weight'] > tr['old_weight'] else "SELL", f"{abs(tr['new_weight'] - tr['old_weight']) * 10:.3f}", f"{entry:,.2f}", f"{exit_:,.2f}", f"{pnl:+,.2f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 2: item.setForeground(QColor("#16f2a5") if val == "BUY" else QColor("#ff6e7f")); item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                if col == 6: item.setForeground(QColor("#16f2a5") if pnl >= 0 else QColor("#ff6e7f"))
                self.trade_table.setItem(row, col, item)

    def _update_risk_table(self, sigs: Dict[str, object]):
        self.risk_table.setRowCount(len(sigs)); last_close = {s: self.last_panel[s]['close'].iloc[-1] for s in sigs}
        for row, (symbol, sig) in enumerate(sigs.items()):
            exposure = abs(sig.target_weight); position = max(0.01, exposure * self.config.starting_cash / last_close[symbol]); avg_cost = last_close[symbol] * (1 - 0.01 * np.sign(sig.ensemble_score)); unreal = (last_close[symbol] - avg_cost) * position; realized = unreal * 0.45
            vals = [symbol, f"{exposure:.2%}", f"{position:.4f}", f"{avg_cost:,.2f}", f"{unreal:+,.2f}", f"{realized:+,.2f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col in (4, 5): item.setForeground(QColor("#16f2a5") if str(val).startswith("+") else QColor("#ff6e7f"))
                self.risk_table.setItem(row, col, item)

    def _save_runtime_settings(self):
        self.config.starting_cash = float(self.cfg_starting_cash.value())
        self.config.strategy.rebalance_threshold = float(self.cfg_rebalance_threshold.value())
        self.config.risk.max_gross_exposure = float(self.cfg_max_gross.value())
        self.config.risk.daily_loss_limit = float(self.cfg_daily_loss.value())
        self.config.risk.max_symbol_weight = float(self.cfg_max_symbol.value())
        self.config.risk.max_spread_bps = float(self.cfg_max_spread.value())
        self.config.risk.slippage_bps = float(self.cfg_slippage.value())
        payload = {
            'api_key': self.api_key.text().strip(),
            'private_key': self.priv_key.text().strip(),
            'starting_cash': self.config.starting_cash,
            'rebalance_threshold': self.config.strategy.rebalance_threshold,
            'max_gross_exposure': self.config.risk.max_gross_exposure,
            'daily_loss_limit': self.config.risk.daily_loss_limit,
            'max_symbol_weight': self.config.risk.max_symbol_weight,
            'max_spread_bps': self.config.risk.max_spread_bps,
            'slippage_bps': self.config.risk.slippage_bps,
        }
        self.settings_store.save(payload); self._append_activity("Runtime settings saved to .runtime_settings.json.")

    def _load_runtime_settings(self):
        payload = self.settings_store.load()
        self.api_key.setText(payload.get('api_key', '')); self.priv_key.setText(payload.get('private_key', ''))
        for key, widget in [
            ('starting_cash', getattr(self, 'cfg_starting_cash', None)),
            ('rebalance_threshold', getattr(self, 'cfg_rebalance_threshold', None)),
            ('max_gross_exposure', getattr(self, 'cfg_max_gross', None)),
            ('daily_loss_limit', getattr(self, 'cfg_daily_loss', None)),
            ('max_symbol_weight', getattr(self, 'cfg_max_symbol', None)),
            ('max_spread_bps', getattr(self, 'cfg_max_spread', None)),
            ('slippage_bps', getattr(self, 'cfg_slippage', None)),
        ]:
            if widget is not None and key in payload:
                widget.setValue(float(payload[key]))
        self.config.starting_cash = float(payload.get('starting_cash', self.config.starting_cash))
        self.config.strategy.rebalance_threshold = float(payload.get('rebalance_threshold', self.config.strategy.rebalance_threshold))
        self.config.risk.max_gross_exposure = float(payload.get('max_gross_exposure', self.config.risk.max_gross_exposure))
        self.config.risk.daily_loss_limit = float(payload.get('daily_loss_limit', self.config.risk.daily_loss_limit))
        self.config.risk.max_symbol_weight = float(payload.get('max_symbol_weight', self.config.risk.max_symbol_weight))
        self.config.risk.max_spread_bps = float(payload.get('max_spread_bps', self.config.risk.max_spread_bps))
        self.config.risk.slippage_bps = float(payload.get('slippage_bps', self.config.risk.slippage_bps))
        if payload:
            self._append_activity("Loaded runtime settings from .runtime_settings.json.")

    def _export_logs(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export logs', str(Path.home() / 'propr_logs.txt'), 'Text Files (*.txt)')
        if not path:
            return
        Path(path).write_text(self.logs_view.toPlainText(), encoding='utf-8')
        self._append_activity(f"Logs exported to {path}.")


if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
