import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import traceback

PLOT_TYPE_META = {
    "折线图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "散点图":    {"needs_y": True,  "needs_x_numeric_or_date": True,  "predictable": True,  "desc": "连续型/时序数据"},
    "条形图":    {"needs_y": True,  "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
    "直方图":    {"needs_y": False, "needs_x_numeric": True,          "predictable": False, "desc": "数值型x"},
    "饼图":      {"needs_y": False, "needs_x_any": True,              "predictable": False, "desc": "分类型/数值型x"},
}

def infer_col_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    try:
        pd.to_datetime(series.dropna()[:10])
        if series.dropna().apply(lambda x: str(x)).str.match(r'^\d{4}[-/年]').sum() > 0:
            return "date"
    except Exception:
        pass
    try:
        pd.to_numeric(series.dropna()[:10])
        return "numeric"
    except Exception:
        pass
    nunique = series.nunique(dropna=True)
    total = series.dropna().size
    if nunique < max(10, total // 4):
        return "category"
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
    lines = []
    if formula_type == "Prophet":
        lines.append(r"$\mathbf{y(t) = g(t) + s(t) + \varepsilon_t}$")
        expl = [
            f"g(t)：趋势项 (类型: {params.get('trend_type', 'N/A')})",
            f"s(t)：季节项 (启用: {params.get('seasonality', 'N/A')})",
            "ε：残差噪声",
            f"MSE: {params.get('mse', float('nan')):.3f}",
            "模型: Prophet"
        ]
        lines += expl
        if all(k in params for k in ['g_vals', 's_vals', 'eps_vals', 'ds_vals']):
            # 用 smart_date_fmt 处理日期
            df_table = pd.DataFrame({
                "日期": [smart_date_fmt(x) for x in params['ds_vals']],
                "g(t)": np.round(params['g_vals'], 3),
                "s(t)": np.round(params['s_vals'], 3),
                "ε": np.round(params['eps_vals'], 3)
            })
            lines.append("")
            lines.append(df_table.to_string(index=False))
    elif formula_type == "线性回归":
        lines.append(r"$\mathbf{y = a x + b}$")
        expl = [
            f"a (斜率): {params.get('a', float('nan')):.3f}",
            f"b (截距): {params.get('b', float('nan')):.3f}",
            f"MSE(均方误差): {params.get('mse', float('nan')):.3f}",
            "模型: 线性回归"
        ]
        lines += expl
        if all(k in params for k in ['x_vals', 'y_true', 'y_pred', 'resid', 'x_label']):
            df_table = pd.DataFrame({
                params['x_label']: [smart_date_fmt(x) for x in params['x_vals']],
                "观测值": np.round(params['y_true'], 3),
                "预测值": np.round(params['y_pred'], 3),
                "残差": np.round(params['resid'], 3)
            })
            lines.append("")
            lines.append(df_table.to_string(index=False))
    elif formula_type == "简单复制":
        lines.append(r"$\mathbf{y = last}$")
        expl = [
            f"last: {params.get('last', float('nan')):.3f}",
            f"MSE: {params.get('mse', float('nan')):.3f}",
            "模型: 简单复制"
        ]
        lines += expl
        if all(k in params for k in ['x_vals', 'y_true', 'y_pred', 'resid', 'x_label']):
            df_table = pd.DataFrame({
                params['x_label']: [smart_date_fmt(x) for x in params['x_vals']],
                "观测值": np.round(params['y_true'], 3),
                "预测值": np.round(params['y_pred'], 3),
                "残差": np.round(params['resid'], 3)
            })
            lines.append("")
            lines.append(df_table.to_string(index=False))
    else:
        lines.append("未知模型")
    return lines
    
class GroupValueFilterBlock(tk.Frame):
    def __init__(self, master, group_field, group_values, on_change, preselect=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.group_field = group_field
        self.vars = {}
        self.on_change = on_change
        self.group_values = group_values
        label = tk.Label(self, text=f"{group_field}分组值:", font=('Microsoft YaHei', 11))
        label.pack(anchor="w", pady=(2, 0))
        btn_frame = tk.Frame(self)
        btn_frame.pack(anchor="w", pady=(0, 4))
        self.all_btn = tk.Button(btn_frame, text="全选", width=7, command=self.select_all, font=('Microsoft YaHei', 10))
        self.all_btn.pack(side=tk.LEFT, padx=(0, 2))
        self.none_btn = tk.Button(btn_frame, text="全不选", width=7, command=self.select_none, font=('Microsoft YaHei', 10))
        self.none_btn.pack(side=tk.LEFT)
        canvas_height = 120
        cb_canvas = tk.Canvas(self, width=140, height=canvas_height, highlightthickness=0)
        cb_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        cb_scrollbar = tk.Scrollbar(self, orient="vertical", command=cb_canvas.yview)
        cb_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        cb_canvas.configure(yscrollcommand=cb_scrollbar.set)
        self.chk_inner = tk.Frame(cb_canvas)
        self.chk_inner.bind("<Configure>", lambda e: cb_canvas.configure(scrollregion=cb_canvas.bbox("all")))
        cb_canvas.create_window((0,0), window=self.chk_inner, anchor="nw")
        pad_width = max([len(str(v)) for v in group_values]) if group_values else 4
        pad_width = min(max(pad_width + 2, 8), 20)
        for v in group_values:
            var = tk.BooleanVar(value=True if (preselect is None or v in preselect) else False)
            cb = tk.Checkbutton(self.chk_inner, text=str(v), variable=var, command=self._changed, anchor="w", width=pad_width, font=('Microsoft YaHei', 10))
            cb.pack(anchor="w", padx=2, pady=1)
            self.vars[v] = var
    def _changed(self):
        selected = [k for k, v in self.vars.items() if v.get()]
        self.on_change(self.group_field, selected)
    def select_all(self):
        for k in self.vars.keys():
            self.vars[k].set(True)
        self._changed()
    def select_none(self):
        for k in self.vars.keys():
            self.vars[k].set(False)
        self._changed()
    def get_selected(self):
        return [k for k, v in self.vars.items() if v.get()]
    def set_selected(self, selected_list):
        for k in self.vars.keys():
            self.vars[k].set(k in selected_list)
        self._changed()

def save_fig_event(event, fig):
    filepath = filedialog.asksaveasfilename(
        title="保存图片",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
    )
    if filepath:
        fig.savefig(filepath)

def make_canvas_with_menu(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    widget = canvas.get_tk_widget()
    menu = tk.Menu(widget, tearoff=False)
    menu.add_command(label="保存图片...", command=lambda: save_fig_event(None, fig))
    def popup_menu(event):
        menu.tk_popup(event.x_root, event.y_root)
    widget.bind("<Button-3>", popup_menu)
    return canvas

class TimeSeriesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数据分析与可视化工具")
        self.df = None
        self.col_types = {}

        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # --- 文件名显示行 ---
        self.filename_var = tk.StringVar(value="未选择文件")
        self.filename_label = tk.Label(top_frame, textvariable=self.filename_var, font=('Microsoft YaHei', 10), fg="#555", anchor="w")
        self.filename_label.grid(row=0, column=1, columnspan=7, sticky="we", padx=(0, 8), pady=(0, 2))
        top_frame.grid_rowconfigure(0, minsize=26)  # 让文件名行有点高度

        # 选择CSV文件按钮占满其格子
        self.file_btn = tk.Button(top_frame, text="选择CSV文件", command=self.load_file, width=14, font=('Microsoft YaHei', 11))
        self.file_btn.grid(row=1, column=0, padx=(0, 8), rowspan=2, sticky="nsew")
        top_frame.grid_columnconfigure(0, weight=1)

        # --- 控件布局调整，原有控件整体下移一行 ---
        self.x_label = tk.Label(top_frame, text="X轴：", font=('Microsoft YaHei', 11))
        self.x_label.grid(row=1, column=1, sticky="w")
        self.x_cb = ttk.Combobox(top_frame, state='readonly')
        self.x_cb.grid(row=1, column=2, padx=(0, 6), sticky="w")
        self.x_type_label = tk.Label(top_frame, text="类型：", font=('Microsoft YaHei', 11))
        self.x_type_label.grid(row=1, column=3, sticky="w")
        self.x_type_var = tk.StringVar(value="自动")
        self.x_type_cb = ttk.Combobox(top_frame, state='readonly', values=["自动", "数字", "日期", "分类型", "文本"], textvariable=self.x_type_var)
        self.x_type_cb.grid(row=1, column=4, padx=(0, 8), sticky="w")
        self.x_cb.bind('<<ComboboxSelected>>', self.on_x_col_changed)
        self.x_type_cb.bind('<<ComboboxSelected>>', self.on_x_type_changed)

        self.y_label = tk.Label(top_frame, text="Y轴：", font=('Microsoft YaHei', 11))
        self.y_label.grid(row=2, column=1, sticky="w")
        self.y_cb = ttk.Combobox(top_frame, state='readonly')
        self.y_cb.grid(row=2, column=2, padx=(0, 8), sticky="w")

        self.plot_type_label = tk.Label(top_frame, text="图类型：", font=('Microsoft YaHei', 11))
        self.plot_type_label.grid(row=1, column=5, sticky="w")
        self.plot_type_var = tk.StringVar(value="散点图")
        self.plot_type_cb = ttk.Combobox(top_frame, state='readonly', values=list(PLOT_TYPE_META.keys()), textvariable=self.plot_type_var)
        self.plot_type_cb.grid(row=1, column=6, padx=(0, 8), sticky="w", pady=(4, 4))
        self.plot_type_cb.bind('<<ComboboxSelected>>', self.on_plot_type_changed)

        self.group_label = tk.Label(top_frame, text="分组字段：", font=('Microsoft YaHei', 11))
        self.group_label.grid(row=1, column=7, rowspan=2, sticky="n")
        self.group_lb_frame = tk.Frame(top_frame)
        self.group_lb_frame.grid(row=1, column=8, rowspan=2, padx=(0, 8), sticky="w")
        self.group_lb_scrollbar = tk.Scrollbar(self.group_lb_frame, orient=tk.VERTICAL)
        self.group_lb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=(4, 4))
        self.group_lb_xscrollbar = tk.Scrollbar(self.group_lb_frame, orient=tk.HORIZONTAL)
        self.group_lb_xscrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=(2, 2), pady=(0, 2))
        self.group_lb = tk.Listbox(
            self.group_lb_frame,
            selectmode=tk.MULTIPLE,
            height=5,
            exportselection=False,
            yscrollcommand=self.group_lb_scrollbar.set,
            xscrollcommand=self.group_lb_xscrollbar.set,
            font=('Microsoft YaHei', 11)
        )
        self.group_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0), pady=(4, 4))
        self.group_lb_scrollbar.config(command=self.group_lb.yview)
        self.group_lb_xscrollbar.config(command=self.group_lb.xview)

        self.sep_top = tk.Frame(main_frame, height=3, bg="#ff2e2e")
        self.sep_top.pack(fill=tk.X, pady=(0, 2))
        self.gblock_canvas = tk.Canvas(main_frame, height=180, highlightthickness=0)
        self.gblock_canvas.pack(fill=tk.X, expand=False)
        self.gblock_scrollbar = tk.Scrollbar(main_frame, orient="horizontal", command=self.gblock_canvas.xview)
        self.gblock_scrollbar.pack(fill=tk.X)
        self.gblock_canvas.configure(xscrollcommand=self.gblock_scrollbar.set)
        self.gblock_frame = tk.Frame(self.gblock_canvas)
        self.gblock_canvas.create_window((0, 0), window=self.gblock_frame, anchor="nw")
        self.gblock_frame.bind("<Configure>", lambda e: self.gblock_canvas.configure(scrollregion=self.gblock_canvas.bbox("all")))
        self.sep_bottom = tk.Frame(main_frame, height=3, bg="#3c7cff")
        self.sep_bottom.pack(fill=tk.X, pady=(2, 0))

        param_btn_frame = tk.Frame(main_frame)
        param_btn_frame.pack(fill=tk.X, pady=3)
        self.dup_label = tk.Label(param_btn_frame, text="重复数据：", font=('Microsoft YaHei', 11))
        self.dup_label.grid(row=0, column=0, sticky="e")
        self.dup_method = tk.StringVar(value="平均")
        self.dup_cb = ttk.Combobox(
            param_btn_frame,
            state='readonly',
            values=["不做更改", "平均", "最大", "最小"],
            textvariable=self.dup_method,
            width=8
        )
        self.dup_cb.grid(row=0, column=1, padx=(0, 8))
        self.step_label = tk.Label(param_btn_frame, text="预测未来步数：", font=('Microsoft YaHei', 11))
        self.step_label.grid(row=0, column=2)
        self.step_var = tk.IntVar(value=7)
        self.step_entry = tk.Entry(param_btn_frame, textvariable=self.step_var, width=5, font=('Microsoft YaHei', 11))
        self.step_entry.grid(row=0, column=3, padx=(0, 8))
        self.model_label = tk.Label(param_btn_frame, text="预测模型：", font=('Microsoft YaHei', 11))
        self.model_label.grid(row=0, column=4)
        self.model_var = tk.StringVar(value="自动")
        self.model_cb = ttk.Combobox(
            param_btn_frame,
            state='readonly',
            values=["自动", "Prophet", "线性回归"],
            textvariable=self.model_var, width=10
        )
        self.model_cb.grid(row=0, column=5, padx=(0, 8))
        self.predict_btn = tk.Button(param_btn_frame, text="画图/预测", command=self.paint_or_predict, width=16, font=('Microsoft YaHei', 11))
        self.predict_btn.grid(row=0, column=6, padx=(10, 0))

        scroll_frame = tk.Frame(main_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.canvas_outer = tk.Canvas(scroll_frame, borderwidth=0, background="#eeeeee", height=600)
        self.canvas_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas_outer.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_outer.configure(yscrollcommand=self.scrollbar.set)
        self.inner_frame = tk.Frame(self.canvas_outer, background="#eeeeee")
        self.canvas_outer.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas_outer.bind_all("<MouseWheel>", self._on_mousewheel)

        self.group_figures = {}
        self.group_canvases = {}
        self.group_value_filter_blocks = {}

        self.group_lb.bind('<<ListboxSelect>>', self.on_group_fields_changed)
        self.x_cb.bind('<<ComboboxSelected>>', self.on_x_col_changed)
        self.plot_type_cb.bind('<<ComboboxSelected>>', self.on_plot_type_changed)

        self.group_value_filters = {}
        self.all_group_values = {}

    def _on_frame_configure(self, event):
        self.canvas_outer.configure(scrollregion=self.canvas_outer.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas_outer.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_file(self):
        filepath = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.filename_var.set(filepath.split("/")[-1] if "/" in filepath else filepath.split("\\")[-1])
            self.df = pd.read_csv(filepath)
            cols = self.df.columns.tolist()
            self.col_types = {c: infer_col_type(self.df[c]) for c in cols}
            self.x_cb['values'] = cols
            self.x_cb.set(cols[0] if cols else "")
            # 动态设置X轴宽度
            maxlen = max([len(str(s)) for s in cols] + [2])
            self.x_cb.config(width=maxlen+2)
            self.x_type_cb.set("自动")
            self.y_cb['values'] = cols
            self.y_cb.set(cols[1] if len(cols) > 1 else (cols[0] if cols else ""))
            self.y_cb.config(width=maxlen+2)
            self.group_lb.delete(0, tk.END)
            for c in cols:
                self.group_lb.insert(tk.END, c)
            # 设置分组字段Listbox宽度
            self.group_lb.config(width=maxlen+2)
            self.refresh_group_fields()
            self.refresh_y_choices()
            self.refresh_param_visibility()
    
    def try_parse_date(self, s):
        if pd.isnull(s):
            return pd.NaT
        str_s = str(s)
        try:
            if "年" in str_s and "月" in str_s:
                return pd.to_datetime(str_s, format='%Y年%m月')
            elif "年" in str_s:
                return pd.to_datetime(str_s, format='%Y年')
            elif "-" in str_s or "/" in str_s:
                return pd.to_datetime(str_s)
            else:
                return pd.to_datetime(str_s, format='%Y')
        except Exception:
            return pd.to_datetime(str_s, errors='coerce')

    def auto_best_formula(self, history_x, history_y):
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

    def on_group_value_filter_change(self, group_field, selected_values):
        if selected_values:
            self.group_value_filters[group_field] = selected_values
        else:
            if group_field in self.group_value_filters:
                del self.group_value_filters[group_field]

    def on_group_fields_changed(self, event=None):
        self.refresh_group_fields()
    
    def refresh_group_fields(self):
        old_selection = dict(self.group_value_filters)
        for widget in self.gblock_frame.winfo_children():
            widget.destroy()
        self.group_value_filter_blocks.clear()
        self.all_group_values.clear()
        group_indices = self.group_lb.curselection()
        group_cols = [self.group_lb.get(i) for i in group_indices]
        if not group_cols or self.df is None:
            self.group_value_filters.clear()
            return
        for group_field in group_cols:
            vals = sorted(self.df[group_field].dropna().unique())
            self.all_group_values[group_field] = vals
            preselect = old_selection.get(group_field, vals)
            block = GroupValueFilterBlock(
                self.gblock_frame,
                group_field, vals,
                self.on_group_value_filter_change,
                preselect=preselect
            )
            block.pack(side=tk.LEFT, padx=10, pady=7, anchor="n")
            self.group_value_filter_blocks[group_field] = block
            self.group_value_filters[group_field] = list(block.get_selected())
        for field in list(self.group_value_filters):
            if field not in group_cols:
                del self.group_value_filters[field]

    def get_selected_group_value_combinations(self, group_cols):
        if not group_cols:
            return [()]
        selected_lists = []
        for g in group_cols:
            vals = self.group_value_filters.get(g)
            if vals:
                selected_lists.append(vals)
            else:
                selected_lists.append([])
        combos = []
        import itertools
        for tup in itertools.product(*selected_lists):
            combos.append(tup)
        return combos

    def clear_inner_frame(self):
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.group_figures.clear()
        self.group_canvases.clear()

    def _handle_duplicates(self, data, x_col, y_col):
        method = self.dup_method.get()
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

    def on_x_col_changed(self, event=None):
        if self.df is None:
            return
        xcol = self.x_cb.get()
        if not xcol:
            return
        t = infer_col_type(self.df[xcol])
        self.col_types[xcol] = t
        self.x_type_cb.set("自动")
        self.refresh_y_choices()
        self.refresh_param_visibility()
    
    def on_x_type_changed(self, event=None):
        self.refresh_y_choices()
        self.refresh_param_visibility()
    
    def on_plot_type_changed(self, event=None):
        self.refresh_y_choices()
        self.refresh_param_visibility()
        if self.group_figures:
            self.paint_or_predict(redraw_only=True)

    def refresh_y_choices(self):
        if self.df is None:
            return
        plot_type = self.plot_type_var.get()
        meta = PLOT_TYPE_META[plot_type]
        cols = self.df.columns.tolist()
        xcol = self.x_cb.get()
        y_choices = cols.copy()
        if meta["needs_y"]:
            if plot_type in ("直方图", "饼图"):
                self.y_cb['values'] = []
                self.y_cb.set("")
                self.y_cb.config(width=8)
                return
            if xcol in y_choices:
                y_choices.remove(xcol)
            y_choices = [c for c in y_choices if infer_col_type(self.df[c])=="numeric"]
            if not y_choices:
                self.y_cb['values'] = []
                self.y_cb.set("")
                self.y_cb.config(width=8)
                return
            self.y_cb['values'] = y_choices
            y_now = self.y_cb.get()
            if y_now not in y_choices:
                self.y_cb.set(y_choices[0])
            # 动态设置Y轴宽度
            maxlen = max([len(str(s)) for s in y_choices] + [2])
            self.y_cb.config(width=maxlen+2)
        else:
            self.y_cb['values'] = []
            self.y_cb.set("")
            self.y_cb.config(width=8)
    
    def refresh_param_visibility(self):
        plot_type = self.plot_type_var.get()
        meta = PLOT_TYPE_META[plot_type]
        xcol = self.x_cb.get()
        x_type = self.get_x_type()
        predictable = meta.get("predictable", False) and (x_type in ("numeric", "date"))
        if predictable:
            self.step_label.grid()
            self.step_entry.grid()
            self.model_label.grid()
            self.model_cb.grid()
            self.predict_btn.config(text="画图/预测", state="normal")
        else:
            self.step_label.grid_remove()
            self.step_entry.grid_remove()
            self.model_label.grid_remove()
            self.model_cb.grid_remove()
            self.predict_btn.config(text="仅画图", state="normal")
    
    def get_x_type(self):
        if self.df is None:
            return ""
        xcol = self.x_cb.get()
        if not xcol:
            return ""
        manual = self.x_type_var.get()
        if manual and manual != "自动":
            return {"数字": "numeric", "日期": "date", "分类型": "category", "文本": "text"}.get(manual, "text")
        return infer_col_type(self.df[xcol])

    def paint_or_predict(self, redraw_only=False):
        try:
            if self.df is None:
                messagebox.showwarning("警告", "请先加载CSV文件！")
                return
            x_col = self.x_cb.get()
            plot_type = self.plot_type_var.get()
            meta = PLOT_TYPE_META[plot_type]
            x_type = self.get_x_type()
            y_col = self.y_cb.get() if meta["needs_y"] else None
            n_periods = self.step_var.get()
            chosen_model = self.model_var.get()
            group_indices = self.group_lb.curselection()
            group_cols = [self.group_lb.get(i) for i in group_indices]
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

                    if show_formula:
                        fig, (ax, ax_formula) = plt.subplots(
                            1, 2,
                            gridspec_kw={'width_ratios': [3, 2]},  # 右侧宽度加大
                            figsize=(16, 5)  # 整体更宽
                        )
                    else:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax_formula = None

                    group_str = f"分组: {group_key_disp}" if group_key_disp else ""

                    # 主图
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
                        try:
                            if model_name == "Prophet":
                                trend_type = best_params.get('trend_type', 'linear')
                                seasonality = best_params.get('seasonality', 'yearly')
                                mse = best_params.get('mse', float('nan'))

                                # prophet预测
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

                                # 提取g(t), s(t), εt数值
                                show_n = min(7, n_periods)
                                g_vals = forecast['trend'][-show_n:].round(3).tolist()
                                if 'seasonal' in forecast.columns:
                                    s_vals = forecast['seasonal'][-show_n:].round(3).tolist()
                                elif 'yearly' in forecast.columns:
                                    s_vals = forecast['yearly'][-show_n:].round(3).tolist()
                                else:
                                    s_vals = [0] * show_n
                                eps_vals = (forecast['yhat'][-show_n:] - forecast['trend'][-show_n:] - (forecast['seasonal'][-show_n:] if 'seasonal' in forecast.columns else 0)).round(3).tolist()
                                ds_vals = forecast['ds'][-show_n:].dt.strftime('%Y-%m-%d').tolist()

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
                                # 未来
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
                                # 构造表格参数（仅历史区间，未来不显示）
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
                                    'x_vals': list(data[x_col]),
                                    'y_true': history_y,
                                    'y_pred': [last_val]*len(history_y),
                                    'resid': resid
                                }
                        except Exception as pred_e:
                            formula_type = "未知"
                            formula_params = {}

                        if ax_formula is not None:
                            ax_formula.axis('off')
                            formula_lines = generate_formula_block(formula_type, formula_params)
                            try:
                                ax_formula.text(
                                    0.03, 0.97, "\n".join(formula_lines),
                                    fontsize=11,  # 字体小一点
                                    ha='left', va='top', color='navy', wrap=True
                                )
                                ax_formula.set_xlim(0, 1.2)  # 拉宽显示区域
                            except Exception as latex_e:
                                ax_formula.text(
                                    0.03, 0.97, "\n".join([str(l) for l in formula_lines]),
                                    fontsize=11, ha='left', va='top', color='navy', wrap=True
                                )
                                ax_formula.set_xlim(0, 1.2)
                                
                    if plot_type not in ("饼图",):
                        ax.set_title(f"{(y_col or '')} vs {x_col} {group_str}")
                        ax.legend()
                    fig.tight_layout(rect=[0, 0, 0.98, 1])
                    fig.subplots_adjust(right=0.98, wspace=0.25)
                    canvas = make_canvas_with_menu(self.inner_frame, fig)
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    self.group_figures[group_key_disp] = fig
                    self.group_canvases[group_key_disp] = canvas
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    continue
            self.inner_frame.update_idletasks()
            self.canvas_outer.configure(scrollregion=self.canvas_outer.bbox("all"))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            messagebox.showerror("出错", f"分析出现异常: {e}\n请查看控制台输出(debug日志)获取详细信息。")
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesPredictorApp(root)
    root.mainloop()