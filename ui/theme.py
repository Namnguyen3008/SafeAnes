"""Design tokens and the application stylesheet.

A slate dark theme: the waveform field stays pure black like a real monitor,
and everything around it sits a few steps lighter so the traces own the screen.

Channel colours follow bedside-monitor convention - ECG green, SpO2 cyan,
arterial pressure red, respiration yellow - so anyone who has worked with a
monitor reads the screen without a legend.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont

# --- surfaces -----------------------------------------------------------------
BG = "#0B1120"
SURFACE = "#0F172A"
CARD = "#172033"
CARD_HOVER = "#1E293B"
INSET = "#0B1222"
BORDER = "#24304A"
BORDER_STRONG = "#3B4A63"
TRACE_BG = "#000000"
TILE_BG = "#05080F"

# --- text ---------------------------------------------------------------------
TEXT = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
TEXT_FAINT = "#64748B"

# --- semantic -----------------------------------------------------------------
ACCENT = "#22C55E"
ACCENT_DIM = "#166534"
DANGER = "#EF4444"
DANGER_DIM = "#450A0A"
WARNING = "#F59E0B"
WARNING_DIM = "#422006"
INFO = "#38BDF8"
FOCUS = "#E2E8F0"

# --- channels -----------------------------------------------------------------
ECG_COLOR = "#39FF14"
PLETH_COLOR = "#22D3EE"
ABP_COLOR = "#FF4D5E"
RESP_COLOR = "#FACC15"

UI_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Inter",
               "Helvetica Neue", "Arial"]
NUMERIC_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "SF Pro Display",
                    "Inter", "Helvetica Neue", "Arial"]


def ui_font(point_size: float = 9.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(UI_FAMILIES)
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


def numeric_font(pixel_size: int, weight: QFont.Weight = QFont.Weight.DemiBold) -> QFont:
    """Large numerics with tabular figures, so a changing value does not jitter."""
    font = QFont()
    font.setFamilies(NUMERIC_FAMILIES)
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    try:                                    # OpenType features arrived in Qt 6.7
        font.setFeature(QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError):
        pass
    return font


STYLESHEET = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QToolTip {{
    background: {CARD_HOVER}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    padding: 5px 7px; border-radius: 4px;
}}

/* -- header ------------------------------------------------------------- */
QFrame#header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QLabel#brand {{ font-size: 15px; font-weight: 600; letter-spacing: 0.3px; }}
QLabel#brandSub {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#clock {{ color: {TEXT_MUTED}; font-size: 13px; }}
QLabel#chip {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 11px;
    padding: 2px 10px; color: {TEXT_MUTED}; font-size: 11px;
}}
QLabel#chip[state="live"] {{ color: {ACCENT}; border-color: {ACCENT_DIM}; }}
QLabel#chip[state="frozen"] {{ color: {INFO}; border-color: #0C4A6E; }}
QLabel#chip[state="scenario"] {{ color: {WARNING}; border-color: {WARNING_DIM}; }}

/* -- side panel ---------------------------------------------------------- */
QWidget#panel, QScrollArea#panel {{ background: {SURFACE}; border: none; }}
QScrollArea#panel > QWidget > QWidget {{ background: {SURFACE}; }}
QLabel#status {{ color: {TEXT_MUTED}; font-size: 11px; padding: 2px 2px; }}

QTabWidget::pane {{ border: none; top: -1px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_FAINT}; border: none;
    padding: 7px 9px; margin-right: 2px; border-radius: 6px; font-size: 11px;
}}
QTabBar::tab:hover {{ color: {TEXT}; background: {CARD}; }}
QTabBar::tab:selected {{ color: {TEXT}; background: {CARD_HOVER}; }}

QFrame#card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
}}
QLabel#cardTitle {{
    color: {TEXT_MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
}}
QLabel#controlName {{ color: #CBD5E1; font-size: 12px; }}
QLabel#controlValue {{ color: {TEXT}; font-size: 12px; font-weight: 600; }}
QLabel#hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#summary {{ color: {TEXT_MUTED}; font-size: 11px; }}

/* -- inputs ------------------------------------------------------------------ */
QSlider {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}
QSlider:focus {{ outline: none; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {ACCENT}; }}

QComboBox, QSpinBox {{
    background: {INSET}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 5px 8px; min-height: 20px; font-size: 12px;
}}
QComboBox:hover, QSpinBox:hover {{ border-color: {BORDER_STRONG}; }}
QComboBox:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_DIM}; outline: none; padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; }}

/* -- buttons ----------------------------------------------------------------- */
QPushButton {{
    background: {CARD_HOVER}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    border-radius: 8px; padding: 8px 12px; font-size: 12px; min-height: 18px;
}}
QPushButton:hover {{ background: #26334A; }}
QPushButton:pressed {{ background: #2D3B55; }}
QPushButton:focus {{ border: 1px solid {FOCUS}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QPushButton#danger {{ background: #7F1D1D; border-color: {DANGER}; font-weight: 600; }}
QPushButton#danger:hover {{ background: #991B1B; }}
QPushButton#warning {{ background: #78350F; border-color: {WARNING}; font-weight: 600; }}
QPushButton#warning:hover {{ background: #92400E; }}
QPushButton#ghost {{ background: transparent; border-color: {BORDER}; color: {TEXT_MUTED}; }}
QPushButton#ghost:hover {{ background: {CARD}; color: {TEXT}; }}
QPushButton#ghost:checked {{ background: #0C4A6E; border-color: {INFO}; color: {TEXT}; }}
QPushButton#segment {{
    background: {INSET}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; color: {TEXT_MUTED}; font-size: 11px;
}}
QPushButton#segment:checked {{ background: {CARD_HOVER}; color: {TEXT}; border-color: {ACCENT}; }}
QPushButton#cpr:checked {{ background: #7C2D12; border-color: #FB923C; color: {TEXT}; }}

QPushButton#rail {{
    background: {SURFACE}; color: {TEXT_MUTED}; border: none; border-radius: 0;
    border-right: 1px solid {BORDER}; padding: 0; font-size: 15px;
}}
QPushButton#rail:hover {{ background: {CARD}; color: {ACCENT}; }}

/* -- lists, progress ------------------------------------------------------- */
QListWidget {{
    background: {INSET}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 4px; outline: none; font-size: 11px;
}}
QListWidget::item {{ padding: 5px 4px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {CARD_HOVER}; color: {TEXT}; }}
QListWidget::item:hover {{ background: {CARD}; }}
QProgressBar {{
    background: {INSET}; border: 1px solid {BORDER}; border-radius: 4px;
    height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {WARNING}; border-radius: 3px; }}

/* -- scrollbars ------------------------------------------------------------ */
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #52627C; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* -- alarm banner ------------------------------------------------------------ */
QLabel#alarm {{ font-size: 13px; font-weight: 600; letter-spacing: 0.5px; padding: 8px 14px; }}
"""
