import matplotlib
import matplotlib.font_manager as fm
import os
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from main_window import TimeSeriesPredictorApp

def set_best_font():
    # 优先尝试常见的中日韩字体，自动检测系统可用字体
    preferred_fonts = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'Noto Sans CJK SC',
        'Noto Sans CJK JP', 'Noto Sans CJK KR', 'WenQuanYi Zen Hei', 'PingFang SC',
        'Source Han Sans SC', 'Source Han Sans CN', 'DejaVu Sans', 'Arial', 'Liberation Sans'
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in preferred_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"[INFO] Using font: {font}")
            return font
    # fallback: use system default
    print("[WARN] No preferred CJK font found, using default font.")
    return None

def main():
    set_best_font()
    app = QApplication(sys.argv)
    font = QFont('Microsoft YaHei', 11)
    app.setFont(font)
    win = TimeSeriesPredictorApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()