import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
import math
import os
import seaborn as sns
import warnings
import traceback
import io
import re
from matplotlib.figure import Figure
from statsmodels.stats.stattools import durbin_watson
# Dynamically select a supported Qt backend and import FigureCanvas
_backend_candidates = [
    ('qtagg', 'matplotlib.backends.backend_qtagg', 'FigureCanvasQTAgg'),
    ('qt5agg', 'matplotlib.backends.backend_qt5agg', 'FigureCanvasQTAgg'),
    ('qt5cairo', 'matplotlib.backends.backend_qt5cairo', 'FigureCanvasQTCairo'),
]
_backend_set = False
for backend, mod, canvas_cls in _backend_candidates:
    try:
        matplotlib.use(backend)
        _FigureCanvasModule = __import__(mod, fromlist=[canvas_cls])
        FigureCanvas = getattr(_FigureCanvasModule, canvas_cls)
        _backend_set = True
        break
    except Exception:
        continue
if not _backend_set:
    warnings.warn("No supported Qt backend found for matplotlib. Some features may not work.")
    FigureCanvas = None
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem, QScrollArea, QSpinBox, QLineEdit, QFileDialog, QMessageBox, QMenu, QTableWidget, QTableWidgetItem, QAbstractItemView, QCheckBox, QFrame, QDialog, QDialogButtonBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QFontMetrics
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

PLOT_TYPE_META = {
    "折线图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "散点图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "条形图":    {"needs_y": True,  "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
    "直方图":    {"needs_y": False, "needs_x_numeric": True,          "predictable": False, "desc": "数值型x"},
    "饼图":      {"needs_y": False, "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
}

def infer_col_type(series):
    
    sample = series.dropna()[:10].astype(str)
    # 先尝试数字型
    try:
        pd.to_numeric(sample)
        return "numeric"
    except Exception:
        pass
    # 再尝试日期型
    try:
        date_like = sample.str.contains(r"^(\d{4}$|\d{4}[-/年]|[-/年]\d{1,2}[-/日])", regex=True).sum() > 0
        if date_like:
            return "date"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "date"
    except Exception:
        pass
    # 再尝试分类型
    nunique = series.nunique(dropna=True)
    total = series.dropna().size
    if nunique < max(10, total // 4):
        return "category"
    # 最后文本
    return "text"

def smart_date_fmt(date_val):
    if isinstance(date_val, pd.Timestamp):
        y, m, d = date_val.year, date_val.month, date_val.day
        if (m == 1 and d == 1) or (m == 12 and d == 31):
            return f"{y}"
        elif d == 1:
            return f"{y}-{m:02d}"
        else:
            return f"{y}-{m:02d}-{d:02d}"
    elif isinstance(date_val, str):
        if len(date_val) == 4 and date_val.isdigit():
            return date_val
        elif date_val.count('-') == 2 and (date_val.split('-')[1] == "01" and date_val.split('-')[2] == "01" or (date_val.split('-')[1] == "12" and date_val.split('-')[2] == "31")):
            return date_val[:4]
        else:
            return date_val
    else:
        return str(date_val)

def generate_formula_block(formula_type, params):
    formula_str = ""
    table_lines = []
    formula_under = ""
    if formula_type == "Prophet":
        formula_str = r"$\mathbf{y(t) = g(t) + s(t) + \varepsilon_t}$"
        formula_under = [
            f"g(t)：趋势项 (类型: {params.get('trend_type', 'N/A')})",
            f"s(t)：季节项 (启用: {params.get('seasonality', 'N/A')})",
            "ε：残差噪声",
            f"MSE: {params.get('mse', float('nan')):.3f}",
            "模型: Prophet"
        ]
        if all(k in params for k in ['g_vals', 's_vals', 'eps_vals', 'ds_vals']):
            df_table = pd.DataFrame({
                "日期": [smart_date_fmt(x) for x in params['ds_vals']],
                "g(t)": np.round(params['g_vals'], 3),
                "s(t)": np.round(params['s_vals'], 3),
                "ε": np.round(params['eps_vals'], 3)
            })
            table_lines.append("")
            table_lines.append(df_table.to_string(index=False))
    elif formula_type == "线性回归":
        formula_str = r"$\mathbf{y = a x + b}$"
        formula_under = [
            f"a (斜率): {params.get('a', float('nan')):.3f}",
            f"b (截距): {params.get('b', float('nan')):.3f}",
            f"MSE(均方误差): {params.get('mse', float('nan')):.3f}",
            "模型: 线性回归"
        ]
        if all(k in params for k in ['x_vals', 'y_true', 'y_pred', 'resid', 'x_label']):
            df_table = pd.DataFrame({
                params['x_label']: [smart_date_fmt(x) for x in params['x_vals']],
                "观测值": np.round(params['y_true'], 3),
                "预测值": np.round(params['y_pred'], 3),
                "残差": np.round(params['resid'], 3)
            })
            table_lines.append("")
            table_lines.append(df_table.to_string(index=False))
    elif formula_type == "多项式拟合":
        degree = params.get('degree', 2)
        coefs = params.get('coefs', [])
        mse = params.get('mse', float('nan'))
        terms = []
        for i, c in enumerate(coefs):
            power = degree - i
            c_ = round(c, 3)
            if power == 0:
                terms.append(f"{c_}")
            elif power == 1:
                terms.append(f"{c_}x")
            else:
                terms.append(f"{c_}x^{power}")
        formula_str = r"$y = " + " + ".join(terms) + r"$"
        formula_under = [
            f"MSE(均方误差): {mse:.3f}",
            f"模型: 多项式拟合 (阶数: {degree})"
        ]
        if all(k in params for k in ['x_vals', 'y_true', 'y_pred', 'resid', 'x_label']):
            df_table = pd.DataFrame({
                params['x_label']: [smart_date_fmt(x) for x in params['x_vals']],
                "观测值": np.round(params['y_true'], 3),
                "预测值": np.round(params['y_pred'], 3),
                "残差": np.round(params['resid'], 3)
            })
            table_lines.append("")
            table_lines.append(df_table.to_string(index=False))
    else:
        formula_str = "未知模型"
    return formula_str, table_lines, formula_under

class GroupValueFilterBlock(QWidget):
    changed = pyqtSignal(str, list)
    def __init__(self, group_field, group_values, preselect=None, parent=None):
        super().__init__(parent)
        self.group_field = group_field
        self.vars = {}
        self.group_values = group_values
        layout = QVBoxLayout(self)
        label = QLabel(f"{group_field}分组值:")
        label.setFixedHeight(20)  # 更矮的标签
        layout.addWidget(label)
        btn_layout = QHBoxLayout()
        self.all_btn = QPushButton("全选")
        self.all_btn.setFixedHeight(24)  # 更矮的按钮
        self.all_btn.clicked.connect(self.select_all)
        btn_layout.addWidget(self.all_btn)
        self.none_btn = QPushButton("全不选")
        self.none_btn.setFixedHeight(24)  # 更矮的按钮
        self.none_btn.clicked.connect(self.select_none)
        btn_layout.addWidget(self.none_btn)
        layout.addLayout(btn_layout)
        self.cb_area = QWidget()
        self.cb_layout = QVBoxLayout(self.cb_area)
        for v in group_values:
            var = QCheckBox(str(v))
            var.setChecked(True if (preselect is None or v in preselect) else False)
            var.stateChanged.connect(self._changed)
            self.cb_layout.addWidget(var)
            self.vars[v] = var
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.cb_area)
        scroll.setFixedHeight(180)  # 选项区更高
        layout.addWidget(scroll)
    def _changed(self):
        selected = [k for k, v in self.vars.items() if v.isChecked()]
        self.changed.emit(self.group_field, selected)
    def select_all(self):
        for v in self.vars.values():
            v.blockSignals(True)
            v.setChecked(True)
            v.blockSignals(False)
    def select_none(self):
        for v in self.vars.values():
            v.blockSignals(True)
            v.setChecked(False)
            v.blockSignals(False)
    def get_selected(self):
        return [k for k, v in self.vars.items() if v.isChecked()]
    def set_selected(self, selected_list):
        for k, v in self.vars.items():
            v.blockSignals(True)
            v.setChecked(k in selected_list)
            v.blockSignals(False)

class DatasetInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据集输入")
        self.selected_type = None
        self.selected_file = None
        self.selected_sheet = None
        self.sheets = []
        layout = QVBoxLayout(self)
        self.type_cb = QComboBox()
        self.type_cb.addItems(["自动检测", "CSV", "Excel(xlsx)", "JSON", "XML"])
        layout.addWidget(QLabel("选择数据类型："))
        layout.addWidget(self.type_cb)
        self.file_btn = QPushButton("选择文件")
        self.file_btn.clicked.connect(self.choose_file)
        layout.addWidget(self.file_btn)
        self.file_label = QLabel("未选择文件")
        layout.addWidget(self.file_label)
        self.sheet_label = QLabel("选择Sheet：")
        self.sheet_cb = QComboBox()
        self.sheet_label.setVisible(False)
        self.sheet_cb.setVisible(False)
        layout.addWidget(self.sheet_label)
        layout.addWidget(self.sheet_cb)
        self.sheet_cb.currentIndexChanged.connect(self.on_sheet_changed)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "所有支持 (*.csv *.xlsx *.xls *.json *.xml);;CSV (*.csv);;Excel (*.xlsx *.xls);;JSON (*.json);;XML (*.xml)")
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            # 检查是否为Excel
            ext = os.path.splitext(file_path)[-1].lower()
            if self.type_cb.currentText() == "Excel(xlsx)" or ext in [".xlsx", ".xls"]:
                try:
                    xls = pd.ExcelFile(file_path)
                    self.sheets = xls.sheet_names
                    self.sheet_cb.clear()
                    self.sheet_cb.addItems(self.sheets)
                    self.sheet_cb.setCurrentIndex(0)
                    self.sheet_label.setVisible(True)
                    self.sheet_cb.setVisible(True)
                except Exception:
                    self.sheets = []
                    self.sheet_label.setVisible(False)
                    self.sheet_cb.setVisible(False)
            else:
                self.sheets = []
                self.sheet_label.setVisible(False)
                self.sheet_cb.setVisible(False)
    def on_sheet_changed(self, idx):
        if self.sheets:
            self.selected_sheet = self.sheets[idx]
    def get_result(self):
        return {
            'type': self.type_cb.currentText(),
            'file': self.selected_file,
            'sheet': self.sheet_cb.currentText() if self.sheet_cb.isVisible() else None
        }

class HeatmapConfigDialog(QDialog):
    def __init__(self, columns, parent=None, preselect_cols=None):
        super().__init__(parent)
        self.setWindowTitle("热力图参数设置")
        layout = QVBoxLayout(self)
        # 选择列（多选列表）
        layout.addWidget(QLabel("请选择要分析的数值列："))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in columns:
            self.col_list.addItem(col)
        # 自动选择上次选择的列
        if preselect_cols:
            for i in range(self.col_list.count()):
                item = self.col_list.item(i)
                if item.text() in preselect_cols:
                    item.setSelected(True)
        else:
            self.col_list.selectAll()
        layout.addWidget(self.col_list)
        # 展示方式
        layout.addWidget(QLabel("热力图展示方式："))
        self.triangle_cb = QComboBox()
        self.triangle_cb.addItems(["全部", "下三角", "上三角"])
        layout.addWidget(self.triangle_cb)
        # 生成模式
        layout.addWidget(QLabel("热力图生成模式："))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["中文（有相关性分析）", "不带数字（只有相关性分析）", "仅有数字"])
        layout.addWidget(self.mode_cb)
        # 按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_config(self):
        cols = [item.text() for item in self.col_list.selectedItems()]
        triangle = self.triangle_cb.currentText()
        mode = self.mode_cb.currentText()
        return cols, triangle, mode

class TimeSeriesPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据分析与可视化工具")
        self.df = None
        self.col_types = {}
        self.group_table_texts = {}
        self.group_figures = {}
        self.group_canvases = {}
        self.group_value_filter_blocks = {}
        self.group_value_filters = {}
        self.all_group_values = {}
        self.corr_result_frame = None
        self.last_heatmap_cols = None  # 记住上次热力图选择的列
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        # 顶部控件区整体封装，固定高度
        top_area = QWidget()
        top_area_layout = QVBoxLayout(top_area)
        top_area_layout.setContentsMargins(0, 0, 0, 0)
        top_area_layout.setSpacing(0)
        # 顶部控件区
        top_frame = QFrame()
        top_layout = QGridLayout(top_frame)
        top_area_layout.addWidget(top_frame)
        # 第一行：数据集输入单独一行
        self.file_btn = QPushButton("数据集输入")
        self.file_btn.clicked.connect(self.dataset_input)
        top_layout.addWidget(self.file_btn, 0, 0, 1, 13)
        # 第二行：X轴、X轴类型、重复数据、预测未来步数、预测模型、多项式阶数
        self.x_label = QLabel("X轴(分类标准)：")
        top_layout.addWidget(self.x_label, 1, 0)
        self.x_cb = QComboBox()
        top_layout.addWidget(self.x_cb, 1, 1)
        self.x_type_label = QLabel("X轴类型：")
        top_layout.addWidget(self.x_type_label, 1, 2)
        self.x_type_cb = QComboBox()
        self.x_type_cb.addItems(["自动", "数字", "日期", "分类型", "文本"])
        top_layout.addWidget(self.x_type_cb, 1, 3)
        self.dup_label = QLabel("重复数据：")
        top_layout.addWidget(self.dup_label, 1, 4)
        self.dup_cb = QComboBox()
        self.dup_cb.addItems(["不做更改", "平均", "最大", "最小"])
        # 默认选中'平均'
        idx_mean = self.dup_cb.findText("平均")
        if idx_mean >= 0:
            self.dup_cb.setCurrentIndex(idx_mean)
        top_layout.addWidget(self.dup_cb, 1, 5)
        self.step_label = QLabel("预测未来步数：")
        top_layout.addWidget(self.step_label, 1, 6)
        self.step_entry = QSpinBox()
        self.step_entry.setRange(1, 365)
        self.step_entry.setValue(7)
        top_layout.addWidget(self.step_entry, 1, 7)
        self.model_label = QLabel("预测模型：")
        top_layout.addWidget(self.model_label, 1, 8)
        self.model_cb = QComboBox()
        self.model_cb.addItems(["自动", "Prophet", "线性回归", "多项式拟合"])
        top_layout.addWidget(self.model_cb, 1, 9)
        self.poly_label = QLabel("多项式阶数：")
        top_layout.addWidget(self.poly_label, 1, 10)
        self.poly_spin = QSpinBox()
        self.poly_spin.setRange(2, 8)
        self.poly_spin.setValue(2)
        top_layout.addWidget(self.poly_spin, 1, 11)
        # 第三行：Y轴、图类型、相关性分析方式、相关性分析按钮、画图/预测按钮
        self.y_label = QLabel("Y轴(判定标准)：")
        top_layout.addWidget(self.y_label, 2, 0)
        self.y_cb = QComboBox()
        top_layout.addWidget(self.y_cb, 2, 1)
        self.plot_type_label = QLabel("图类型：")
        top_layout.addWidget(self.plot_type_label, 2, 2)
        self.plot_type_cb = QComboBox()
        self.plot_type_cb.addItems(list(PLOT_TYPE_META.keys()))
        # 默认选中'散点图'
        idx_scatter = self.plot_type_cb.findText("散点图")
        if idx_scatter >= 0:
            self.plot_type_cb.setCurrentIndex(idx_scatter)
        top_layout.addWidget(self.plot_type_cb, 2, 3)
        self.corr_mode_label = QLabel("相关性分析方式：")
        top_layout.addWidget(self.corr_mode_label, 2, 4)
        self.corr_mode_cb = QComboBox()
        self.corr_mode_cb.addItems(["XY", "热力图"])
        top_layout.addWidget(self.corr_mode_cb, 2, 5)
        self.corr_btn = QPushButton("相关性分析")
        top_layout.addWidget(self.corr_btn, 2, 6)
        self.predict_btn = QPushButton("画图/预测")
        top_layout.addWidget(self.predict_btn, 2, 7)
        # 分组字段+分组值筛选区（同一行）
        group_row = QHBoxLayout()
        self.group_label = QLabel("分组字段：")
        group_row.addWidget(self.group_label)
        self.group_lb = QListWidget()
        self.group_lb.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.group_lb.setMaximumWidth(220)
        group_row.addWidget(self.group_lb)
        self.gblock_area = QWidget()
        self.gblock_layout = QHBoxLayout(self.gblock_area)
        self.gblock_area.setLayout(self.gblock_layout)
        self.gblock_scroll = QScrollArea()
        self.gblock_scroll.setWidgetResizable(True)
        self.gblock_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.gblock_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gblock_scroll.setWidget(self.gblock_area)
        self.gblock_scroll.setMinimumHeight(220)
        group_row.addWidget(self.gblock_scroll, stretch=1)
        top_area_layout.addLayout(group_row)
        # 固定顶部控件区高度（可根据实际内容微调高度）
        top_area.setFixedHeight(340)
        main_layout.addWidget(top_area)
        # 展示区
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 设置下方区域高度为顶部一半
        self.scroll_area.setMinimumHeight(600)
        self.inner_frame = QWidget()
        self.inner_layout = QVBoxLayout(self.inner_frame)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)
        self.inner_layout.setSpacing(0)
        # 关键：让inner_frame不拦截鼠标事件，滚轮可传递到scroll_area
        self.inner_frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.scroll_area.setWidget(self.inner_frame)
        main_layout.addWidget(self.scroll_area, stretch=1)
        # 事件绑定
        self.x_cb.currentIndexChanged.connect(self.on_x_col_changed)
        self.x_type_cb.currentIndexChanged.connect(self.on_x_type_changed)
        self.plot_type_cb.currentIndexChanged.connect(self.on_plot_type_changed)
        self.group_lb.itemSelectionChanged.connect(self.on_group_fields_changed)
        self.predict_btn.clicked.connect(self.paint_or_predict)
        self.corr_btn.clicked.connect(self.correlation_analysis)

    def set_combobox_width(self, combobox, items, min_width=8, max_width=40):
        # 计算最长文本宽度，设置合适宽度
        font = combobox.font() if hasattr(combobox, 'font') else QFont('Microsoft YaHei', 11)
        metrics = QFontMetrics(font)
        maxlen = max([metrics.horizontalAdvance(str(s)) for s in items] + [min_width])
        # 额外加点宽度，防止被截断
        combobox.setMinimumWidth(min(maxlen + 30, max_width * 12))

    def try_parse_date(self, s):
        if pd.isnull(s):
            return pd.NaT
        str_s = str(s)
        try:
            if "年" in str_s and "月" in str_s:
                return pd.to_datetime(str_s, format="%Y年%m月%d日", errors='coerce')
            elif "年" in str_s:
                return pd.to_datetime(str_s, format="%Y年", errors='coerce')
            elif "-" in str_s or "/" in str_s:
                return pd.to_datetime(str_s, errors='coerce')
            else:
                return pd.to_datetime(str_s, errors='coerce')
        except Exception:
            return pd.to_datetime(str_s, errors='coerce')

    def _safe_to_datetime(self, series):
        """自动检测日期格式，优先用format参数，否则suppress警告"""
        sample = series.dropna().astype(str).head(10)
        fmt = None
        # 常见格式检测
        if all(re.match(r"^\d{4}-\d{2}-\d{2}$", s) for s in sample):
            fmt = "%Y-%m-%d"
        elif all(re.match(r"^\d{4}/\d{2}/\d{2}$", s) for s in sample):
            fmt = "%Y/%m/%d"
        elif all(re.match(r"^\d{4}年\d{1,2}月\d{1,2}日$", s) for s in sample):
            fmt = "%Y年%m月%d日"
        elif all(re.match(r"^\d{4}$", s) for s in sample):
            fmt = "%Y"
        if fmt:
            return pd.to_datetime(series, format=fmt, errors='coerce')
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                return pd.to_datetime(series, errors='coerce')

    def auto_best_formula(self, history_x, history_y):
        best_model = None
        best_mse = np.inf
        best_params = {}
        try:
            if not np.issubdtype(history_x.dtype, np.datetime64):
                fake_dates = self._safe_to_datetime(history_x.astype(str))
                df_prophet = pd.DataFrame({'ds': fake_dates, 'y': history_y})
            else:
                df_prophet = pd.DataFrame({'ds': history_x, 'y': history_y})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)
            y_pred = model.predict(df_prophet)['yhat']
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_model = "Prophet"
                best_mse = mse
                best_params = {'trend_type': 'linear', 'seasonality': 'yearly', 'mse': mse}
        except Exception:
            pass
        try:
            x_idx = np.arange(len(history_x)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x_idx, history_y)
            y_pred = lr.predict(x_idx)
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_model = "线性回归"
                best_mse = mse
                best_params = {'a': lr.coef_[0], 'b': lr.intercept_, 'mse': mse}
        except Exception:
            pass
        try:
            degree = 2
            X = np.arange(len(history_x))
            coefs = np.polyfit(X, history_y, degree)
            poly = np.poly1d(coefs)
            y_pred = poly(X)
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_model = "多项式拟合"
                best_mse = mse
                best_params = {'degree': degree, 'coefs': coefs, 'mse': mse}
        except Exception:
            pass
        return best_model, best_params

    def on_group_value_filter_change(self, group_field, selected_values):
        self.group_value_filters[group_field] = selected_values
        # 移除自动绘图逻辑，分组值变动不再自动触发绘图
        # self.paint_or_predict()

    def on_group_fields_changed(self):
        self.refresh_group_fields()

    def refresh_group_fields(self):
        # 记住每个分组字段当前已选的分组值
        prev_selected = {g: block.get_selected() for g, block in self.group_value_filter_blocks.items()}
        for i in reversed(range(self.gblock_layout.count())):
            widget = self.gblock_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.group_value_filter_blocks.clear()
        self.all_group_values.clear()
        group_indices = self.group_lb.selectedIndexes()
        group_cols = [self.group_lb.item(i.row()).text() for i in group_indices]
        if not group_cols or self.df is None:
            return
        for group_field in group_cols:
            values = list(self.df[group_field].dropna().unique())
            self.all_group_values[group_field] = values
            block = GroupValueFilterBlock(group_field, values)
            # 恢复之前的选择
            if group_field in prev_selected:
                block.set_selected(prev_selected[group_field])
            block.changed.connect(self.on_group_value_filter_change)
            self.gblock_layout.addWidget(block)
            self.group_value_filter_blocks[group_field] = block
        for field in list(self.group_value_filters):
            if field not in group_cols:
                del self.group_value_filters[field]

    def get_selected_group_value_combinations(self, group_cols):
        if not group_cols:
            return [()]
        selected_lists = []
        for g in group_cols:
            block = self.group_value_filter_blocks.get(g)
            if block:
                selected = block.get_selected()
                if not selected:
                    return []
                selected_lists.append(selected)
            else:
                return []
        combos = list(itertools.product(*selected_lists))
        return combos

    def clear_inner_frame(self):
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.group_figures.clear()
        self.group_canvases.clear()

    def add_canvas_to_inner_frame(self, canvas):
        # 保持canvas原始大小，超出部分可滚动，竖直排列
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        container_layout.addWidget(canvas)
        # 关键：container高度设为canvas的原始高度，防止被压缩
        container.setMinimumHeight(canvas.height())
        container.setMaximumHeight(canvas.height())

        # 已移除右键菜单和弹窗展示相关代码

        self.inner_layout.addWidget(container)
        spacer = QWidget()
        spacer.setFixedHeight(20)
        self.inner_layout.addWidget(spacer)

    def _handle_duplicates(self, data, x_col, y_col):
        method = self.dup_cb.currentText()
        if method == "不做更改":
            return data
        agg_map = {
            "平均": "mean",
            "最大": "max",
            "最小": "min"
        }
        agg_func = agg_map.get(method, None)
        if agg_func:
            if y_col in data:
                data = data.groupby(x_col, as_index=False)[y_col].agg(agg_func)
            else:
                data = data.drop_duplicates(subset=x_col)
        return data

    def on_x_col_changed(self):
        if self.df is None:
            return
        xcol = self.x_cb.currentText()
        if not xcol:
            return
        t = infer_col_type(self.df[xcol])
        self.col_types[xcol] = t
        # self.x_type_cb.setCurrentIndex(0)  # 不再重置X轴类型
        self.refresh_y_choices()
        self.refresh_param_visibility()

    def on_x_type_changed(self):
        self.refresh_y_choices()
        self.refresh_param_visibility()

    def on_plot_type_changed(self):
        self.refresh_y_choices()
        self.refresh_param_visibility()
        if self.group_figures:
            self.paint_or_predict(redraw_only=True)

    def refresh_y_choices(self):
        if self.df is None:
            return
        plot_type = self.plot_type_cb.currentText()
        meta = PLOT_TYPE_META[plot_type]
        cols = self.df.columns.tolist()
        xcol = self.x_cb.currentText()
        y_choices = cols.copy()
        prev_y = self.y_cb.currentText() # 记住当前Y轴
        if meta["needs_y"]:
            if plot_type in ("直方图", "饼图"):
                self.y_cb.clear()
                return
            if xcol in y_choices:
                y_choices.remove(xcol)
            y_choices = [c for c in y_choices if infer_col_type(self.df[c])=="numeric"]
            self.y_cb.clear()
            self.y_cb.addItems(y_choices)
            self.set_combobox_width(self.y_cb, y_choices)
            # 恢复Y轴选择
            if prev_y in y_choices:
                self.y_cb.setCurrentText(prev_y)
            elif y_choices:
                self.y_cb.setCurrentIndex(0)
        else:
            self.y_cb.clear()

    def refresh_param_visibility(self):
        plot_type = self.plot_type_cb.currentText()
        meta = PLOT_TYPE_META[plot_type]
        xcol = self.x_cb.currentText()
        x_type = self.get_x_type()
        predictable = meta.get("predictable", False) and (x_type in ("numeric", "date"))
        self.step_label.setVisible(predictable)
        self.step_entry.setVisible(predictable)
        self.model_label.setVisible(predictable)
        self.model_cb.setVisible(predictable)
        self.poly_label.setVisible(predictable and self.model_cb.currentText() == "多项式拟合")
        self.poly_spin.setVisible(predictable and self.model_cb.currentText() == "多项式拟合")
        self.predict_btn.setText("画图/预测" if predictable else "仅画图")

    def get_x_type(self):
        if self.df is None:
            return ""
        xcol = self.x_cb.currentText()
        if not xcol:
            return ""
        manual = self.x_type_cb.currentText()
        if manual and manual != "自动":
            return {"数字": "numeric", "日期": "date", "分类型": "category", "文本": "text"}.get(manual, "text")
        return infer_col_type(self.df[xcol])

    def paint_or_predict(self, redraw_only=False):
        try:
            if self.df is None:
                QMessageBox.warning(self, "警告", "请先加载CSV文件！")
                return
            x_col = self.x_cb.currentText()
            plot_type = self.plot_type_cb.currentText()
            meta = PLOT_TYPE_META[plot_type]
            x_type = self.get_x_type()
            y_col = self.y_cb.currentText() if meta["needs_y"] else None
            n_periods = self.step_entry.value()
            chosen_model = self.model_cb.currentText()
            group_indices = self.group_lb.selectedIndexes()
            group_cols = [self.group_lb.item(i.row()).text() for i in group_indices]
            combos = self.get_selected_group_value_combinations(group_cols)
            self.clear_inner_frame()
            for combo in combos:
                if not group_cols:
                    group_data = self.df
                    group_key_disp = ()
                else:
                    cond = np.ones(len(self.df), dtype=bool)
                    for i, g in enumerate(group_cols):
                        cond = cond & (self.df[g] == combo[i])
                    group_data = self.df[cond]
                    group_key_disp = tuple(combo)
                if len(group_data) == 0:
                    continue
                try:
                    if meta["needs_y"]:
                        cols_needed = [x_col, y_col]
                    else:
                        cols_needed = [x_col]
                    data = group_data[cols_needed].dropna(subset=[x_col] + ([y_col] if y_col else []))
                    if x_type == "date":
                        data[x_col] = data[x_col].apply(self.try_parse_date)
                        data = data.dropna(subset=[x_col])
                        data = data.sort_values(by=x_col)
                    elif x_type == "numeric":
                        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
                        data = data.dropna(subset=[x_col])
                        data = data.sort_values(by=x_col)
                    data = self._handle_duplicates(data, x_col, y_col) if y_col else data
                    if len(data) == 0:
                        continue
                    show_formula = False
                    formula_type = None
                    formula_params = {}
                    if plot_type in ("折线图", "散点图") and meta.get("predictable") and x_type in ("date", "numeric") and y_col:
                        show_formula = True
                    # 优化自适应figsize逻辑
                    min_width, min_height = 7, 4.5  # 更小的最小宽高
                    base_width = 10  # 基础宽度
                    base_height = 5  # 基础高度
                    xlab_len = max([len(str(x)) for x in data[x_col]]) if len(data) > 0 else 4
                    ylab_len = max([len(str(y)) for y in data[y_col]]) if y_col and len(data) > 0 else 4
                    n_points = len(data)
                    group_str_len = len(str(group_key_disp))
                    # 数据点数少时不额外放大
                    width = base_width + min(xlab_len, 40) * 0.13 + min(group_str_len, 40) * 0.09
                    height = base_height + min(ylab_len, 40) * 0.13
                    # 数据点数多时才逐步放大
                    if n_points > 30:
                        width += min(n_points, 2000) * 0.002
                        height += min(n_points, 2000) * 0.002
                    if plot_type == "饼图" and n_points > 10:
                        height += (n_points - 10) * 0.10
                    width = max(width, min_width)
                    height = max(height, min_height)
                    # 获取展示区最大高度（像素），转为英寸
                    max_pix_height = self.scroll_area.height() if hasattr(self, 'scroll_area') else 700
                    dpi = matplotlib.rcParams.get('figure.dpi', 100)
                    max_inch_height = max_pix_height / dpi
                    # 限制height不超过展示区高度
                    height = min(height, max_inch_height)
                    if show_formula:
                        import matplotlib.gridspec as gridspec
                        fig = matplotlib.pyplot.figure(figsize=(max(width, 12), max(height, 7)))
                        # 使用GridSpec布局，左上(0,0)为主图，右上(0,1)为公式
                        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1.5])
                        ax = fig.add_subplot(gs[0, 0])
                        ax_formula = fig.add_subplot(gs[0, 1])
                    else:
                        fig, ax = matplotlib.pyplot.subplots(figsize=(max(width, 8), max(height, 4.5)))
                        ax_formula = None
                        # ax_table = None
                    group_str = f"分组: {group_key_disp}" if group_key_disp else ""
                    if plot_type == "折线图":
                        ax.plot(data[x_col], data[y_col], marker="o", label="历史")
                    elif plot_type == "散点图":
                        ax.scatter(data[x_col], data[y_col], marker="o", label="历史")
                    elif plot_type == "条形图":
                        ax.bar(data[x_col], data[y_col], label="值", alpha=0.8)
                    elif plot_type == "直方图":
                        ax.hist(data[x_col], bins=20, label=x_col, alpha=0.8)
                    elif plot_type == "饼图":
                        counts = data[x_col].value_counts()
                        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
                        ax.set_title(f"{x_col} 分布 {group_str}")
                    if show_formula and not redraw_only:
                        history_x = data[x_col]
                        history_y = data[y_col].values
                        model_name, best_params = "", {}
                        if chosen_model == "自动":
                            model_name, best_params = self.auto_best_formula(history_x, history_y)
                        else:
                            model_name = chosen_model
                            best_params = {}
                        try:
                            if model_name == "Prophet":
                                trend_type = best_params.get('trend_type', 'linear')
                                seasonality = best_params.get('seasonality', 'yearly')
                                mse = best_params.get('mse', float('nan'))
                                if not np.issubdtype(history_x.dtype, np.datetime64):
                                    fake_dates = self._safe_to_datetime(history_x.astype(str))
                                    df_prophet = pd.DataFrame({'ds': fake_dates, 'y': history_y})
                                else:
                                    df_prophet = pd.DataFrame({'ds': history_x, 'y': history_y})
                                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                                model.fit(df_prophet)
                                freq = pd.infer_freq(df_prophet['ds'])
                                if freq is None:
                                    freq = 'YE'
                                future = model.make_future_dataframe(periods=n_periods, freq=freq)
                                forecast = model.predict(future)
                                pred_dates = forecast["ds"][-n_periods:]
                                pred_values = forecast["yhat"][-n_periods:]
                                ax.plot(forecast['ds'], forecast['yhat'], "--", marker="x", label="预测")
                                show_n = min(7, n_periods)
                                g_vals = forecast['trend'][-show_n:].round(3).tolist()
                                if 'seasonal' in forecast.columns:
                                    s_vals = forecast['seasonal'][-show_n:].round(3).tolist()
                                elif 'yearly' in forecast.columns:
                                    s_vals = forecast['yearly'][-show_n:].round(3).tolist()
                                else:
                                    s_vals = [0] * show_n
                                eps_vals = (forecast['yhat'][-show_n:] - forecast['trend'][-show_n:] - (forecast['seasonal'][-show_n:] if 'seasonal' in forecast.columns else 0)).round(3).tolist()
                                ds_vals = forecast['ds'][-show_n:]
                                formula_type = "Prophet"
                                formula_params = {
                                    'trend_type': trend_type,
                                    'seasonality': seasonality,
                                    'mse': mse,
                                    'g_vals': g_vals,
                                    's_vals': s_vals,
                                    'eps_vals': eps_vals,
                                    'ds_vals': ds_vals
                                }
                            elif model_name == "线性回归":
                                data = data.reset_index(drop=True)
                                X = np.arange(len(data)).reshape(-1, 1)
                                y = data[y_col].values
                                lr = LinearRegression()
                                lr.fit(X, y)
                                X_future = np.arange(len(data), len(data) + n_periods).reshape(-1, 1)
                                y_pred_all = lr.predict(np.vstack([X, X_future]))
                                if x_type == "date":
                                    future_dates = pd.date_range(data[x_col].max(), periods=n_periods+1, freq='YE')[1:] if len(data[x_col]) > 1 else []
                                else:
                                    dx = np.median(np.diff(data[x_col])) if len(data[x_col]) > 1 else 1
                                    future_dates = [data[x_col].max() + dx * (i+1) for i in range(n_periods)]
                                min_len = min(len(future_dates), len(y_pred_all[-n_periods:]))
                                future_dates = future_dates[:min_len]
                                y_pred_future = y_pred_all[-n_periods:][:min_len]
                                ax.plot(list(data[x_col]) + list(future_dates), y_pred_all, "--", marker="x", label="预测", color="red")
                                formula_type = "线性回归"
                                mse = mean_squared_error(y, lr.predict(X))
                                resid = y - lr.predict(X)
                                formula_params = {
                                    'a': lr.coef_[0],
                                    'b': lr.intercept_,
                                    'mse': mse,
                                    'x_label': x_col,
                                    'x_vals': list(data[x_col]),
                                    'y_true': y,
                                    'y_pred': lr.predict(X),
                                    'resid': resid
                                }
                            elif model_name == "多项式拟合":
                                data = data.reset_index(drop=True)
                                if x_type == "date":
                                    X = np.arange(len(data))
                                    y = data[y_col].values
                                    degree = self.poly_spin.value()
                                    coefs = np.polyfit(X, y, degree)
                                    poly = np.poly1d(coefs)
                                    # 预测未来索引
                                    X_future = np.arange(len(data), len(data) + n_periods)
                                    X_all = np.concatenate([X, X_future])
                                    y_pred_all = poly(X_all)
                                    # 历史日期
                                    date_hist = list(data[x_col])
                                    # 未来日期，间隔与历史一致
                                    if len(date_hist) > 1:
                                        freq = pd.infer_freq(pd.Series(date_hist))
                                        if freq is not None:
                                            date_future = list(pd.date_range(date_hist[-1], periods=n_periods+1, freq=freq)[1:])
                                        else:
                                            delta = date_hist[1] - date_hist[0]
                                            date_future = [date_hist[-1] + delta * (i+1) for i in range(n_periods)]
                                    else:
                                        date_future = []
                                    x_plot = date_hist + date_future
                                    ax.plot(x_plot, y_pred_all, "--", marker="x", label="多项式预测", color="orange")
                                else:
                                    X = np.array(data[x_col])
                                    y = data[y_col].values
                                    degree = self.poly_spin.value()
                                    coefs = np.polyfit(X, y, degree)
                                    poly = np.poly1d(coefs)
                                    dx = np.median(np.diff(X)) if len(X) > 1 else 1
                                    X_future = np.array([X.max() + dx * (i+1) for i in range(n_periods)])
                                    X_all = np.concatenate([X, X_future])
                                    y_pred_all = poly(X_all)
                                    ax.plot(X_all, y_pred_all, "--", marker="x", label="多项式预测", color="orange")
                                formula_type = "多项式拟合"
                                mse = mean_squared_error(y, poly(X))
                                resid = y - poly(X)
                                formula_params = {
                                    'degree': degree,
                                    'coefs': coefs,
                                    'mse': mse,
                                    'x_label': x_col,
                                    'x_vals': list(data[x_col]),
                                    'y_true': y,
                                    'y_pred': poly(X),
                                    'resid': resid
                                }
                        except Exception:
                            formula_type = "未知模型"
                            formula_params = {}
                        # if ax_table is not None and y_col is not None:
                        #     # 只显示部分数据，防止过大
                        #     max_rows = 8
                        #     x_vals = list(data[x_col])
                        #     y_vals = list(data[y_col])
                        #     n_rows = min(len(x_vals), max_rows)
                        #     # 横向表格：表头在左侧，数据横排
                        #     cell_text = [x_vals[:n_rows], y_vals[:n_rows]]  # 2行n_rows列
                        #     row_labels = [x_col, y_col]
                        #     col_labels = [str(i+1) for i in range(n_rows)]
                        #     the_table = ax_table.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0,0,1,1])
                        #     the_table.auto_set_font_size(False)
                        #     the_table.set_fontsize(12)
                        #     the_table.scale(1.2, 1.2)
                        #     for (row, col), cell in the_table.get_celld().items():
                        #         cell.set_linewidth(1)
                        #         cell.PAD = 0.3
                        # 公式内容放右侧，使用LaTeX渲染，避免多余$
                        if ax_formula is not None:
                            ax_formula.axis('off')
                            formula_str, table_lines, formula_under = generate_formula_block(formula_type, formula_params)
                            if formula_str:
                                fstr = formula_str.strip()
                                if not (fstr.startswith("$") and fstr.endswith("$")):
                                    fstr = f"${fstr}$"
                                ax_formula.text(0.5, 0.5, fstr, fontsize=16, ha='center', va='center', color='navy', transform=ax_formula.transAxes)
                            if formula_under:
                                ax_formula.text(0.5, 0.2, "\n".join([str(l) for l in formula_under]), fontsize=13, ha='center', va='top', color='red', transform=ax_formula.transAxes)
                    if plot_type not in ("饼图",):
                        ax.set_title(f"{(y_col or '')} vs {x_col} {group_str}")
                        ax.legend()
                    fig.tight_layout(rect=[0, 0, 0.98, 1])
                    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.12, wspace=0.25)  # 增加四周padding
                    canvas = FigureCanvas(fig)
                    # 让canvas自适应scroll_area宽度，超宽可横向滚动
                    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    self.add_canvas_to_inner_frame(canvas)
                    self.group_figures[group_key_disp] = fig
                    self.group_canvases[group_key_disp] = canvas
                except Exception as e:
                    traceback.print_exc()
                    continue
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "出错", f"分析出现异常: {e}\n请查看控制台输出(debug日志)获取详细信息。")
            return

    def correlation_analysis(self):
        try:
            if self.df is None:
                QMessageBox.warning(self, "警告", "请先加载CSV文件！")
                return
            self.clear_inner_frame()
            mode = self.corr_mode_cb.currentText()
            group_indices = self.group_lb.selectedIndexes()
            group_cols = [self.group_lb.item(i.row()).text() for i in group_indices]
            combos = self.get_selected_group_value_combinations(group_cols)
            if not combos:
                combos = [()]
            for combo in combos:
                if not group_cols:
                    df = self.df.copy()
                    group_key_disp = ()
                else:
                    cond = np.ones(len(self.df), dtype=bool)
                    for i, g in enumerate(group_cols):
                        cond = cond & (self.df[g] == combo[i])
                    df = self.df[cond].copy()
                    group_key_disp = tuple(combo)
                if len(df) == 0:
                    continue
                try:
                    if mode == "XY":
                        x_col = self.x_cb.currentText()
                        y_col = self.y_cb.currentText()
                        if not x_col or not y_col or x_col == y_col:
                            QMessageBox.warning(self, "警告", "请选择不同的X、Y列！")
                            return
                        if infer_col_type(df[x_col]) != "numeric" or infer_col_type(df[y_col]) != "numeric":
                            QMessageBox.warning(self, "警告", "X、Y列必须都是数值型，当前选择的列类型不符。")
                            return
                        cols = [x_col, y_col]
                        df = df[cols].dropna()
                        pearson = df.corr(method="pearson")
                        spearman = df.corr(method="spearman")
                        kendall = df.corr(method="kendall")
                        # 线性回归用于AIC/BIC等
                        X = df[[x_col]].values
                        y = df[y_col].values
                        lr = LinearRegression().fit(X, y)
                        y_pred = lr.predict(X)
                        resid = y - y_pred
                        n = len(y)
                        k = 2  # 截距+斜率
                        rss = np.sum(resid ** 2)  # 残差平方和
                        tss = np.sum((y - np.mean(y)) ** 2)  # 总平方和
                        ess = np.sum((y_pred - np.mean(y)) ** 2)  # 解释平方和
                        r2 = lr.score(X, y)
                        try:
                            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else float('nan')
                        except Exception:
                            adj_r2 = float('nan')
                        # 标准误、t统计量
                        s2 = rss / (n - k) if n > k else float('nan')
                        try:
                            X_design = np.column_stack([np.ones(n), X.flatten()])
                            cov = s2 * np.linalg.inv(X_design.T @ X_design)
                            se = np.sqrt(np.diag(cov))  # [截距, 斜率]
                        except Exception:
                            se = [float('nan'), float('nan')]
                        t_intercept = lr.intercept_ / se[0] if se[0] != 0 else float('nan')
                        t_slope = lr.coef_[0] / se[1] if se[1] != 0 else float('nan')
                        f_stat = (ess / (k - 1)) / (rss / (n - k)) if (n > k and rss > 0) else float('nan')
                        # Durbin-Watson
                        dw = durbin_watson(resid)
                        # AIC/BIC
                        aic = n * math.log(rss / n) + 2 * k if n > 0 and rss > 0 else float('nan')
                        bic = n * math.log(rss / n) + k * math.log(n) if n > 0 and rss > 0 else float('nan')
                        # 相关性类型
                        r = pearson.iloc[0,1]
                        if abs(r) < 0.2:
                            rel_type = "无相关"
                        elif r > 0:
                            rel_type = "正相关"
                        else:
                            rel_type = "负相关"
                        # 自相关
                        autocorr = "是" if x_col == y_col else "否"
                        row_labels = [
                            "变量1", "变量2", "皮尔逊r", "斯皮尔曼r", "肯德尔r", "AIC", "BIC", "相关性类型", "自相关",
                            "残差平方和(RSS)", "总平方和(TSS)", "解释平方和(ESS)",
                            "决定系数R²", "调整后R²", "截距标准误", "斜率标准误", "截距t统计量", "斜率t统计量", "F统计量", "Durbin-Watson"
                        ]
                        col_labels = ["统计量", "值"]
                        table_data = [
                            ["变量1", x_col],
                            ["变量2", y_col],
                            ["皮尔逊r", f"{pearson.iloc[0,1]:.4f}"],
                            ["斯皮尔曼r", f"{spearman.iloc[0,1]:.4f}"],
                            ["肯德尔r", f"{kendall.iloc[0,1]:.4f}"],
                            ["AIC", f"{aic:.2f}"],
                            ["BIC", f"{bic:.2f}"],
                            ["相关性类型", rel_type],
                            ["自相关", autocorr],
                            ["残差平方和(RSS)", f"{rss:.4f}"],
                            ["总平方和(TSS)", f"{tss:.4f}"],
                            ["解释平方和(ESS)", f"{ess:.4f}"],
                            ["决定系数R²", f"{r2:.4f}"],
                            ["调整后R²", f"{adj_r2:.4f}"],
                            ["截距标准误", f"{se[0]:.4f}"],
                            ["斜率标准误", f"{se[1]:.4f}"],
                            ["截距t统计量", f"{t_intercept:.4f}"],
                            ["斜率t统计量", f"{t_slope:.4f}"],
                            ["F统计量", f"{f_stat:.4f}"],
                            ["Durbin-Watson", f"{dw:.4f}"]
                        ]
                        fig, ax = matplotlib.pyplot.subplots(figsize=(6.5, 13))
                        ax.axis('off')
                        # 添加分组标题
                        if group_key_disp:
                            title_str = f"分组: {group_key_disp}"
                        else:
                            title_str = "全部数据"
                        ax.set_title(title_str, fontsize=15, pad=18)
                        the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', rowLabels=None, bbox=[0,0,1,1])
                        the_table.auto_set_font_size(False)
                        the_table.set_fontsize(12)
                        the_table.scale(1.5, 2.0)
                        # 三线表美化
                        nrows = len(table_data)
                        ncols = len(col_labels)
                        for (row, col), cell in the_table.get_celld().items():
                            cell.set_fontsize(12)
                            cell.PAD = 0.35
                            # 上边线
                            if row == 0:
                                cell.set_linewidth(2)
                            # 下边线
                            if row == nrows-1:
                                cell.set_linewidth(2)
                            # 表头下边线
                            if row == 0 and col < ncols:
                                cell.set_linewidth(2)
                        # 表头加粗
                        for col in range(ncols):
                            the_table[0, col].set_fontsize(13)
                            the_table[0, col].set_text_props(weight='bold')
                        canvas = FigureCanvas(fig)
                        self.add_canvas_to_inner_frame(canvas)
                    elif mode == "热力图":
                        num_cols = [c for c in df.columns if infer_col_type(df[c]) == "numeric"]
                        if len(num_cols) < 2:
                            QMessageBox.warning(self, "警告", "可用的数值型列不足2个，无法生成热力图！")
                            continue
                        # 记忆上次选择的列
                        dlg = HeatmapConfigDialog(num_cols, self, preselect_cols=self.last_heatmap_cols)
                        if dlg.exec() != QDialog.DialogCode.Accepted:
                            return  # 取消后立即return，防止后续执行
                        sel_cols, triangle, mode_str = dlg.get_config()
                        if len(sel_cols) < 2:
                            QMessageBox.warning(self, "警告", "请选择至少2个数值型列！")
                            continue
                        self.last_heatmap_cols = sel_cols  # 记住本次选择
                        corr = df[sel_cols].corr(method="pearson")
                        mask = None
                        if triangle == "下三角":
                            mask = np.triu(np.ones_like(corr, dtype=bool), 1)
                        elif triangle == "上三角":
                            mask = np.tril(np.ones_like(corr, dtype=bool), -1)
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        figsize = (min(8, 1+len(sel_cols)), min(6, 1+len(sel_cols)))
                        fig, ax = plt.subplots(figsize=figsize)
                        annot = True if mode_str in ("中文（有相关性分析）", "仅有数字") else False
                        fmt = ".2f"
                        cmap = "coolwarm"
                        if mask is not None:
                            sns.heatmap(corr, annot=annot, mask=mask, cmap=cmap, ax=ax, fmt=fmt, square=True, cbar=True)
                        else:
                            sns.heatmap(corr, annot=annot, cmap=cmap, ax=ax, fmt=fmt, square=True, cbar=True)                        # 分组信息
                        group_str = f"分组: {group_key_disp}" if group_key_disp else ""
                        if mode_str == "中文（有相关性分析）":
                            ax.set_title(f"皮尔逊相关性热力图（中文） {group_str}")
                        elif mode_str == "不带数字（只有相关性分析）":
                            ax.set_title(f"皮尔逊相关性热力图（无数字） {group_str}")
                        else:
                            ax.set_title(f"皮尔逊相关性热力图（仅数字） {group_str}")
                        if mode_str == "仅有数字":
                            for _, spine in ax.spines.items():
                                spine.set_visible(False)
                            ax.collections.clear()
                        canvas = FigureCanvas(fig)
                        self.add_canvas_to_inner_frame(canvas)
                except Exception as e:
                    traceback.print_exc()
                    QMessageBox.critical(self, "分析出错", f"分组{group_key_disp}分析异常: {e}\n请查看控制台输出(debug日志)获取详细信息。")
                    continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "出错", f"分析出现异常: {e}\n请查看控制台输出(debug日志)获取详细信息。")
            return

    def dataset_input(self):
        dlg = DatasetInputDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            result = dlg.get_result()
            file_path = result['file']
            dtype = result['type']
            sheet = result['sheet']
            if not file_path:
                QMessageBox.warning(self, "警告", "未选择文件！")
                return
            ext = os.path.splitext(file_path)[-1].lower()
            try:
                if dtype == "CSV" or (dtype == "自动检测" and ext == ".csv"):
                    df = pd.read_csv(file_path)
                elif dtype == "Excel(xlsx)" or (dtype == "自动检测" and ext in [".xlsx", ".xls"]):
                    if sheet:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                    else:
                        df = pd.read_excel(file_path)
                elif dtype == "JSON" or (dtype == "自动检测" and ext == ".json"):
                    df = pd.read_json(file_path)
                elif dtype == "XML" or (dtype == "自动检测" and ext == ".xml"):
                    df = pd.read_xml(file_path)
                else:
                    QMessageBox.warning(self, "警告", f"暂不支持的文件类型: {ext}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "出错", f"读取数据失败: {e}")
                return
            self.setWindowTitle(f"数据分析与可视化工具 -- 已载入 {os.path.basename(file_path)}")
            self.df = df
            cols = self.df.columns.tolist()
            self.col_types = {c: infer_col_type(self.df[c]) for c in cols}
            self.x_cb.clear()
            self.x_cb.addItems(cols)
            self.set_combobox_width(self.x_cb, cols)
            if cols:
                self.x_cb.setCurrentIndex(0)
            self.x_type_cb.setCurrentIndex(0)
            self.y_cb.clear()
            self.y_cb.addItems(cols)
            self.set_combobox_width(self.y_cb, cols)
            if len(cols) > 1:
                self.y_cb.setCurrentIndex(1)
            elif cols:
                self.y_cb.setCurrentIndex(0)
            self.group_lb.clear()
            for c in cols:
                self.group_lb.addItem(c)
            self.group_lb.setMinimumWidth(min(max([len(str(s)) for s in cols] + [8]) * 12, 40 * 12))
            self.refresh_group_fields()
            self.refresh_y_choices()
            self.refresh_param_visibility()
            self.last_heatmap_cols = None  # 数据源变更时重置热力图列记忆

def main():
    app = QApplication(sys.argv)
    font = QFont('Microsoft YaHei', 11)
    app.setFont(font)
    win = TimeSeriesPredictorApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()