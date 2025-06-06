import sys
import os
import math
import traceback
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
from PyQt6.QtWidgets import (QMenu, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, QComboBox, QPushButton, QSpinBox, QListWidget, QAbstractItemView, QSizePolicy, QScrollArea, QMessageBox, QDialog)
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from prophet import Prophet
from utils import infer_col_type, generate_formula_block, smart_date_fmt
from group_value_filter_block import GroupValueFilterBlock
from numeric_filter_dialog import NumericFilterDialog
from heatmap_config_dialog import HeatmapConfigDialog
from dataset_input_dialog import DatasetInputDialog

PLOT_TYPE_META = {
    "折线图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "散点图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "条形图":    {"needs_y": True,  "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
    "直方图":    {"needs_y": False, "needs_x_numeric": True,          "predictable": False, "desc": "数值型x"},
    "饼图":      {"needs_y": False, "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
}

# 这里将 TimeSeriesPredictorApp 主窗口类迁移至此文件
# 请在后续步骤中补充完整实现

class TimeSeriesPredictorApp(QMainWindow):
    # ...完整实现从 main.py 迁移，此处省略，见上一轮内容...
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
        self.y_label = QLabel("Y轴(判定标准)：")
        top_layout.addWidget(self.y_label, 2, 0)
        self.y_cb = QComboBox()
        top_layout.addWidget(self.y_cb, 2, 1)
        self.plot_type_label = QLabel("图类型：")
        top_layout.addWidget(self.plot_type_label, 2, 2)
        self.plot_type_cb = QComboBox()
        self.plot_type_cb.addItems(list(PLOT_TYPE_META.keys()))
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
        group_row = QHBoxLayout()
        self.group_label = QLabel("分组字段：")
        group_row.addWidget(self.group_label)
        self.group_lb = QListWidget()
        self.group_lb.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.group_lb.setMaximumWidth(220)
        group_row.addWidget(self.group_lb)
        self.gblock_area = QWidget()
        self.gblock_layout = QHBoxLayout(self.gblock_area)
        self.gblock_layout.setContentsMargins(0, 0, 0, 0)
        self.gblock_layout.setSpacing(8)
        self.gblock_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.gblock_area.setLayout(self.gblock_layout)
        self.gblock_scroll = QScrollArea()
        self.gblock_scroll.setWidgetResizable(True)
        self.gblock_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.gblock_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gblock_scroll.setWidget(self.gblock_area)
        self.gblock_scroll.setMinimumHeight(220)
        self.gblock_scroll.setMaximumHeight(220)
        group_row.addWidget(self.gblock_scroll, stretch=1)
        top_area_layout.addLayout(group_row)
        top_area.setFixedHeight(340)
        main_layout.addWidget(top_area)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setMinimumHeight(600)
        self.inner_frame = QWidget()
        self.inner_layout = QVBoxLayout(self.inner_frame)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)
        self.inner_layout.setSpacing(0)
        self.inner_frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.scroll_area.setWidget(self.inner_frame)
        main_layout.addWidget(self.scroll_area, stretch=1)
        self.x_cb.currentIndexChanged.connect(self.on_x_col_changed)
        self.x_type_cb.currentIndexChanged.connect(self.on_x_type_changed)
        self.plot_type_cb.currentIndexChanged.connect(self.on_plot_type_changed)
        self.group_lb.itemSelectionChanged.connect(self.on_group_fields_changed)
        self.predict_btn.clicked.connect(self.paint_or_predict)
        self.corr_btn.clicked.connect(self.correlation_analysis)

    def dataset_input(self):
        dialog = DatasetInputDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            dtype = result['type']
            file_path = result['file']
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

    def on_x_col_changed(self):
        if self.df is None:
            return
        xcol = self.x_cb.currentText()
        if not xcol:
            return
        t = infer_col_type(self.df[xcol])
        self.col_types[xcol] = t
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
        prev_y = self.y_cb.currentText()
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

    def on_group_fields_changed(self):
        self.refresh_group_fields()

    def refresh_group_fields(self):
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
            col_type = infer_col_type(self.df[group_field])
            block = GroupValueFilterBlock(group_field, values, col_type)
            if group_field in prev_selected:
                block.set_selected(prev_selected[group_field])
            block.changed.connect(self.on_group_value_filter_change)
            self.gblock_layout.addWidget(block)
            self.group_value_filter_blocks[group_field] = block
        for field in list(self.group_value_filters):
            if field not in group_cols:
                self.group_value_filters.pop(field)

    def get_selected_group_value_combinations(self, group_cols):
        if not group_cols:
            return [()]
        selected_lists = []
        for g in group_cols:
            block = self.group_value_filter_blocks.get(g)
            if block is None:
                selected_lists.append([None])
                continue
            sel = block.get_selected()
            if isinstance(sel, dict) and ("advanced" in sel or "mode" in sel):
                selected_lists.append([sel])
            else:
                selected_lists.append(sel if sel else [None])
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
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        container_layout.addWidget(canvas)
        container.setMinimumHeight(canvas.height())
        container.setMaximumHeight(canvas.height())
        self.inner_layout.addWidget(container)
        spacer = QWidget()
        spacer.setFixedHeight(20)
        self.inner_layout.addWidget(spacer)

        # 添加右键菜单导出图片功能
        def export_figure():
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(canvas, "导出图片", "", "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*)")
            if file_path:
                try:
                    canvas.figure.savefig(file_path)
                except Exception as e:
                    QMessageBox.critical(canvas, "保存失败", f"保存图片失败: {e}")

        def contextMenuEvent(event):
            menu = QMenu(canvas)
            export_action = menu.addAction("导出图片...")
            export_action.triggered.connect(export_figure)
            menu.exec(event.globalPos())

        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(contextMenuEvent)

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
            # 只按x_col分组，忽略分组字段，保证x唯一
            if y_col in data:
                data = data.groupby(x_col, as_index=False)[y_col].agg(agg_func)
            else:
                data = data.drop_duplicates(subset=x_col)
        return data

    def on_group_value_filter_change(self, group_field, selected_values):
        self.group_value_filters[group_field] = selected_values

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
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            from prophet import Prophet
            from utils import generate_formula_block
            table_lines = []
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
                        # 使用改进的日期解析函数处理各种日期格式，包括纯年份
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

                    # === 自适应figsize逻辑 ===
                    base_width = 14
                    base_height = 8
                    xlab_len = max([len(str(x)) for x in data[x_col]]) if len(data) > 0 else 4
                    ylab_len = max([len(str(y)) for y in data[y_col]]) if y_col and len(data) > 0 else 4
                    n_points = len(data)
                    group_str_len = len(str(group_key_disp))
                    width = base_width + min(xlab_len, 40) * 0.18 + min(group_str_len, 40) * 0.13 + min(n_points, 2000) * 0.003
                    height = base_height + min(ylab_len, 40) * 0.18 + min(n_points, 2000) * 0.003
                    if plot_type == "饼图" and n_points > 10:
                        height += (n_points - 10) * 0.12
                    if show_formula:
                        fig, (ax, ax_formula) = plt.subplots(
                            1, 2,
                            gridspec_kw={'width_ratios': [4, 1.5]},
                            figsize=(max(width, 16), max(height, 9))
                        )
                    else:
                        fig, ax = plt.subplots(figsize=(max(width, 14), max(height, 8)))
                        ax_formula = None

                    group_str = ""                    # 主图
                    if plot_type == "折线图":
                        if x_type == "date":
                            ax.plot(data[x_col], data[y_col], marker="o", label="历史")
                        else:
                            ax.plot(data[x_col], data[y_col], marker="o", label="历史")
                    elif plot_type == "散点图":
                        if x_type == "date":
                            ax.scatter(data[x_col], data[y_col], marker="o", label="历史")
                        else:
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
                                    fake_dates = pd.to_datetime(history_x.astype(str), errors='coerce')
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
                                ds_vals = [smart_date_fmt(x) for x in forecast['ds'][-show_n:]]
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
                                ax.plot(future_dates, y_pred_future, "--", marker="x", label="预测", color="red")
                                formula_type = "线性回归"
                                mse = mean_squared_error(y, lr.predict(X))
                                resid = y - lr.predict(X)
                                formula_params = {
                                    'a': lr.coef_[0],
                                    'b': lr.intercept_,
                                    'mse': mse,
                                    'x_label': x_col,
                                    'x_vals': list(history_x) if x_type == "date" else list(data[x_col]),
                                    'y_true': y,
                                    'y_pred': lr.predict(X),
                                    'resid': resid
                                }
                            elif model_name == "简单复制":
                                last_val = history_y[-1]
                                if x_type == "date":
                                    future_dates = pd.date_range(history_x.max(), periods=n_periods+1, freq='YE')[1:] if len(history_x) > 1 else []
                                else:
                                    dx = np.median(np.diff(data[x_col])) if len(data[x_col]) > 1 else 1
                                    future_dates = [data[x_col].max() + dx * (i+1) for i in range(n_periods)]
                                min_len = min(len(future_dates), n_periods)
                                future_dates = future_dates[:min_len]
                                pred = [last_val] * min_len
                                ax.plot(future_dates, pred, "o--", label="预测", color="red")
                                formula_type = "简单复制"
                                mse = mean_squared_error(history_y, [last_val]*len(history_y))
                                resid = history_y - [last_val]*len(history_y)
                                formula_params = {
                                    'last': last_val,
                                    'mse': mse,
                                    'x_label': x_col,
                                    'x_vals': list(history_x) if x_type == "date" else list(data[x_col]),
                                    'y_true': history_y,
                                    'y_pred': [last_val]*len(history_y),
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
                                    if len(data[x_col]) > 1:
                                        freq = pd.infer_freq(pd.Series(data[x_col]))
                                        if freq is None:
                                            freq = 'D'
                                        future_dates = pd.date_range(data[x_col].max(), periods=n_periods+1, freq=freq)[1:]
                                    else:
                                        future_dates = []
                                    X_future = np.arange(len(data), len(data) + len(future_dates))
                                    X_all = np.concatenate([X, X_future])
                                    x_plot = list(data[x_col]) + list(future_dates)
                                    y_pred_all = poly(X_all)
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
                                    'x_vals': list(history_x) if x_type == "date" else list(data[x_col]),
                                    'y_true': y,
                                    'y_pred': poly(X),
                                    'resid': resid
                                }
                        except Exception as pred_e:
                            formula_type = "未知"
                            formula_params = {}

                        if ax_formula is not None:
                            ax_formula.axis('off')
                            formula_str, table_lines, formula_under = generate_formula_block(formula_type, formula_params)
                            try:
                                ax.text(0.5, -0.18, formula_str, fontsize=14, ha='center', va='top', color='navy', wrap=True, transform=ax.transAxes)
                            except Exception as latex_e:
                                ax.text(0.5, -0.18, str(formula_str), fontsize=14, ha='center', va='top', color='navy', wrap=True, transform=ax.transAxes)
                            ax.text(0.5, -0.3, "\n".join([str(l) for l in formula_under]), fontsize=14, ha='center', va='top', color='red', wrap=True, transform=ax.transAxes)
                            try:
                                ax_formula.text(
                                    0.03, 1, "\n".join(table_lines),
                                    fontsize=10,  # 字体小一点
                                    ha='left', va='top', color='navy', wrap=True
                                )
                                ax_formula.set_xlim(0, 1.2)  # 拉宽显示区域
                            except Exception as latex_e:
                                ax_formula.text(
                                    0.03, 1, "\n".join([str(l) for l in table_lines]),
                                    fontsize=10, ha='left', va='top', color='navy', wrap=True
                                )
                                ax_formula.set_xlim(0, 1.2)
                    if plot_type not in ("饼图",):
                        ax.set_title(f"{(y_col or '')} vs {x_col} {group_str}")
                        ax.legend()
                        ax.margins(x=0.05, y=0.1)  # 自动加边距，防止数据被遮挡
                    fig.tight_layout(rect=[0, 0, 0.98, 1])
                    fig.subplots_adjust(right=0.98, wspace=0.25)
                    canvas = FigureCanvas(fig)
                    self.add_canvas_to_inner_frame(canvas)
                    self.group_figures[group_key_disp] = fig
                    self.group_canvases[group_key_disp] = canvas
                    self.group_table_texts[group_key_disp] = (
                        "\n".join([l if i != 1 else l.replace(' ', '\t') for i, l in enumerate(table_lines)])
                        if 'table_lines' in locals() and table_lines else ""
                    )
                except Exception as e:
                    traceback.print_exc()
                    continue
            self.inner_frame.update()
            self.scroll_area.setWidget(self.inner_frame)
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
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import scipy.stats as stats
            import statsmodels.api as sm
            from statsmodels.stats import stattools as smtools
            import seaborn as sns
            from PyQt6.QtWidgets import QLabel, QMenu, QApplication
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from utils import infer_col_type
            mode = self.corr_mode_cb.currentText() if hasattr(self, 'corr_mode_cb') else 'XY'
            if mode == "热力图":
                dialog = HeatmapConfigDialog(self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                # 可在此处获取dialog配置参数，后续生成热力图
                # ...（热力图生成逻辑待补充）...
                return
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
                if mode == "XY":
                    x_col = self.x_cb.currentText()
                    y_col = self.y_cb.currentText()
                    if not x_col or not y_col or x_col == y_col:
                        label = QLabel("请选择不同的X、Y列！")
                        label.setStyleSheet("color: red; font-size: 14px;")
                        self.inner_layout.addWidget(label)
                        return
                    if infer_col_type(df[x_col]) != "numeric" or infer_col_type(df[y_col]) != "numeric":
                        label = QLabel("X、Y列必须都是数值型，当前选择的列类型不符。")
                        label.setStyleSheet("color: red; font-size: 14px;")
                        self.inner_layout.addWidget(label)
                        return
                    cols = [x_col, y_col]
                    df = df[cols].dropna()
                else:
                    num_cols = [c for c in df.columns if infer_col_type(df[c]) == "numeric"]
                    if len(num_cols) < 2:
                        label = QLabel("数值型列不足2个，无法分析。")
                        label.setStyleSheet("color: red; font-size: 14px;")
                        self.inner_layout.addWidget(label)
                        return
                    cols = num_cols
                    df = df[cols].dropna()
                if group_cols:
                    group_str = ", ".join(f"{g}={v}" for g, v in zip(group_cols, combo))
                    label = QLabel(f"分组: {group_str}")
                    label.setStyleSheet("color: #1a237e; font-weight: bold; font-size: 14px;")
                    self.inner_layout.addWidget(label)
                pearson = df.corr(method="pearson")
                stat_rows = []
                for i, col1 in enumerate(cols):
                    for j, col2 in enumerate(cols):
                        if j <= i:
                            continue
                        x = df[col1]
                        y = df[col2]
                        r, p = stats.pearsonr(x, y)
                        r_s, p_s = stats.spearmanr(x, y)
                        r_k, p_k = stats.kendalltau(x, y)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        y_pred = model.predict(X)
                        resid = y - y_pred
                        ssr = np.sum((y_pred - np.mean(y))**2)
                        sse = np.sum((y - y_pred)**2)
                        sst = np.sum((y - np.mean(y))**2)
                        r2 = model.rsquared
                        r2_adj = model.rsquared_adj
                        std_err = model.bse.iloc[1] if len(model.bse) > 1 else float('nan')
                        t_val = model.tvalues.iloc[1] if len(model.tvalues) > 1 else float('nan')
                        t_p = model.pvalues.iloc[1] if len(model.pvalues) > 1 else float('nan')
                        f_val = model.fvalue
                        f_p = model.f_pvalue
                        aic = model.aic
                        bic = model.bic
                        dw = smtools.durbin_watson(resid)
                        if abs(r) > 0.8:
                            rel = "强相关"
                        elif abs(r) > 0.5:
                            rel = "中等相关"
                        elif abs(r) > 0.3:
                            rel = "弱相关"
                        else:
                            rel = "几乎无相关"
                        rel += ", 正相关" if r > 0 else ", 负相关" if r < 0 else ", 无方向"
                        rel += ", 显著" if p < 0.05 else ", 不显著"
                        if dw < 1.5:
                            dw_str = f"{dw:.2f} (正自相关)"
                        elif dw > 2.5:
                            dw_str = f"{dw:.2f} (负自相关)"
                        else:
                            dw_str = f"{dw:.2f} (无自相关)"
                        stat_rows.append([
                            col1, col2,
                            f"{r:.3f}", f"{p:.3g}", rel,
                            f"{r_s:.3f}", f"{p_s:.3g}",
                            f"{r_k:.3f}", f"{p_k:.3g}",
                            f"{ssr:.2f}", f"{sse:.2f}", f"{sst:.2f}",
                            f"{r2:.3f}", f"{r2_adj:.3f}",
                            f"{std_err:.3g}", f"{t_val:.3g}", f"{t_p:.3g}",
                            f"{f_val:.3g}", f"{f_p:.3g}",
                            f"{aic:.2f}", f"{bic:.2f}",
                            dw_str
                        ])
                # === 主图+四个表格嵌套布局，表格区高度自适应 ===
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                for row in stat_rows:
                    # 1. 计算表格区所需高度（每行高度*行数+表头+间距）
                    table1_rows = 2
                    table2_rows = 2
                    table3_rows = 2
                    table4_rows = 2
                    # 估算每个表格高度（可根据实际内容微调）
                    table_height = 0.7  # 单个表格基础高度
                    table_gap = 0.18    # 表格间距
                    main_height = 3.5   # 主图高度
                    # 总高度 = 主图 + 4表格 + 3间隔
                    total_height = main_height + 4*table_height + 3*table_gap
                    fig = plt.figure(figsize=(13, total_height))
                    gs_main = gridspec.GridSpec(2, 1, height_ratios=[main_height, 4*table_height+3*table_gap])
                    # 主图
                    ax_main = fig.add_subplot(gs_main[0])
                    x = df[cols[0]]
                    y = df[cols[1]]
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    y_pred = model.predict(X)
                    ax_main.scatter(x, y, label="观测值")
                    ax_main.plot(x, y_pred, color="red", label="最小二乘拟合")
                    ax_main.set_xlabel(cols[0])
                    ax_main.set_ylabel(cols[1])
                    # 主标题
                    main_title = f"{cols[1]} ~ {cols[0]} 最小二乘法拟合"
                    # 分组副标题
                    if group_cols:
                        group_str = ", ".join(f"{g}={v}" for g, v in zip(group_cols, combo))
                        ax_main.set_title(main_title, fontsize=15, pad=16, loc='center')
                        ax_main.text(0.5, 1.3, f"分组: {group_str}", fontsize=12, color="#444", ha='center', va='bottom', transform=ax_main.transAxes)
                    else:
                        ax_main.set_title(main_title, fontsize=15, pad=16, loc='center')
                    ax_main.legend()
                    # 表格区嵌套GridSpec
                    gs_tables = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[1], height_ratios=[1,1,1,1], hspace=table_gap/table_height)
                    # 1. 变量1/变量2/相关性分析/WD自相关分析
                    ax1 = fig.add_subplot(gs_tables[0])
                    ax1.axis('off')
                    table1 = [
                        ["变量1", "变量2", "相关性分析", "WD自相关分析"],
                        [row[0], row[1], row[4], row[-1]]
                    ]
                    t1 = ax1.table(cellText=table1, loc='center', cellLoc='center', colWidths=[0.25]*4)
                    t1.auto_set_font_size(False)
                    t1.set_fontsize(12)
                    t1.scale(1.1, 1.2)
                    for c in range(4):
                        t1[0, c].set_fontsize(13)
                        t1[0, c].set_text_props(weight='bold')
                    # 2. 皮尔逊/斯皮尔曼/肯德尔相关性
                    ax2 = fig.add_subplot(gs_tables[1])
                    ax2.axis('off')
                    table2 = [
                        ["皮尔逊r", "皮尔逊可信度", "斯皮尔曼r", "斯皮尔曼可信度", "肯德尔r", "肯德尔可信度"],
                        [row[2], row[3], row[5], row[6], row[7], row[8]]
                    ]
                    t2 = ax2.table(cellText=table2, loc='center', cellLoc='center', colWidths=[1/6]*6)
                    t2.auto_set_font_size(False)
                    t2.set_fontsize(12)
                    t2.scale(1.1, 1.2)
                    for c in range(6):
                        t2[0, c].set_fontsize(13)
                        t2[0, c].set_text_props(weight='bold')
                    # 3. 平方和
                    ax3 = fig.add_subplot(gs_tables[2])
                    ax3.axis('off')
                    table3 = [
                        ["解释平方和", "残差平方和", "总平方和"],
                        [row[9], row[10], row[11]]
                    ]
                    t3 = ax3.table(cellText=table3, loc='center', cellLoc='center', colWidths=[1/3]*3)
                    t3.auto_set_font_size(False)
                    t3.set_fontsize(12)
                    t3.scale(1.1, 1.15)
                    for c in range(3):
                        t3[0, c].set_fontsize(13)
                        t3[0, c].set_text_props(weight='bold')
                    # 4. R2等统计量
                    ax4 = fig.add_subplot(gs_tables[3])
                    ax4.axis('off')
                    table4 = [
                        ["R2", "调整R2", "标准误", "t值", "F值", "F可信度", "AIC", "BIC"],
                        [row[12], row[13], row[14], row[15], row[17], row[18], row[19], row[20]]
                    ]
                    t4 = ax4.table(cellText=table4, loc='center', cellLoc='center', colWidths=[1/8]*8)
                    t4.auto_set_font_size(False)
                    t4.set_fontsize(12)
                    t4.scale(1.1, 1.2)
                    for c in range(8):
                        t4[0, c].set_fontsize(13)
                        t4[0, c].set_text_props(weight='bold')
                    fig.tight_layout(rect=[0, 0, 1, 1])
                    canvas = FigureCanvas(fig)
                    self.add_canvas_to_inner_frame(canvas)
                    canvas.draw()
                    plt.close(fig)
                self.inner_frame.update()
                self.scroll_area.setWidget(self.inner_frame)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "出错", f"相关性分析出现异常: {e}\n请查看控制台输出(debug日志)获取详细信息。")
            return

    def set_combobox_width(self, combobox, items, min_width=8, max_width=40):
        font = combobox.font() if hasattr(combobox, 'font') else QFont('Microsoft YaHei', 11)
        metrics = QFontMetrics(font)
        maxlen = max([metrics.horizontalAdvance(str(s)) for s in items] + [min_width])
        combobox.setMinimumWidth(min(maxlen + 30, max_width * 12))

    def auto_best_formula(self, history_x, history_y):
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from prophet import Prophet
        best_model = None
        best_mse = np.inf
        best_params = {}
        try:
            if not np.issubdtype(history_x.dtype, np.datetime64):
                fake_dates = pd.to_datetime(history_x.astype(str), errors='coerce')
                df_prophet = pd.DataFrame({'ds': fake_dates, 'y': history_y})
            else:
                df_prophet = pd.DataFrame({'ds': history_x, 'y': history_y})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)
            y_pred = model.predict(df_prophet)['yhat']
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_model = "Prophet"
                trend_type = getattr(model, 'trend', 'linear')
                if hasattr(model, 'seasonalities'):
                    seasonality = list(model.seasonalities.keys())
                else:
                    seasonality = []
                best_params = {
                    'trend_type': trend_type,
                    'seasonality': ','.join(seasonality) if seasonality else '无',
                    'mse': mse,
                    'model_obj': model
                }
        except Exception as e:
            print("[DEBUG] Prophet 拟合失败:", e, file=sys.stderr)
        try:
            x_idx = np.arange(len(history_x)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x_idx, history_y)
            y_pred = lr.predict(x_idx)
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_model = "线性回归"
                best_params = {
                    'a': lr.coef_[0],
                    'b': lr.intercept_,
                    'mse': mse,
                }
        except Exception as e:
            print("[DEBUG] 线性回归拟合失败:", e, file=sys.stderr)
        try:
            y_pred = [history_y[-1]] * len(history_y)
            mse = mean_squared_error(history_y, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_model = "简单复制"
                best_params = {
                    'last': history_y[-1],
                    'mse': mse
                }
        except Exception as e:
            print("[DEBUG] 简单复制拟合失败:", e, file=sys.stderr)
        return best_model, best_params

    def try_parse_date(self, s):
        import pandas as pd
        if pd.isnull(s):
            return pd.NaT
        str_s = str(s).strip()
        try:
            # 处理中文年月格式
            if "年" in str_s and "月" in str_s:
                return pd.to_datetime(str_s, format='%Y年%m月')
            elif "年" in str_s:
                return pd.to_datetime(str_s, format='%Y年')
            # 处理纯年份（如2000, 2010, 2004）
            elif str_s.isdigit() and len(str_s) == 4:
                year = int(str_s)
                return pd.Timestamp(year=year, month=1, day=1)
            # 处理带分隔符的日期
            elif "-" in str_s or "/" in str_s:
                return pd.to_datetime(str_s)
            else:
                # 尝试作为年份处理
                try:
                    year = int(float(str_s))  # 处理可能的浮点数格式
                    if 1900 <= year <= 2100:  # 合理的年份范围
                        return pd.Timestamp(year=year, month=1, day=1)
                except:
                    pass
                return pd.to_datetime(str_s, format='%Y')
        except Exception:
            return pd.to_datetime(str_s, errors='coerce')
