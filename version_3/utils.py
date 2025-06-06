import pandas as pd
import numpy as np
import re

def infer_col_type(series):
    sample = series.dropna()[:10].astype(str)
    try:
        pd.to_numeric(sample)
        return "numeric"
    except Exception:
        pass
    try:
        date_like = sample.str.contains(r"^(\d{4}$|\d{4}[-/年]|[-/年]\d{1,2}[-/日])", regex=True).sum() > 0
        if date_like:
            return "date"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "date"
    except Exception:
        pass
    nunique = series.nunique(dropna=True)
    total = series.dropna().size
    if nunique < max(10, total // 4):
        return "category"
    return "text"

def smart_date_fmt(date_val):
    # 处理各种类型的日期数据
    if pd.isnull(date_val):
        return "NaT"
    
    # 如果是pandas Timestamp
    if isinstance(date_val, pd.Timestamp):
        y, m, d = date_val.year, date_val.month, date_val.day
        if (m == 1 and d == 1) or (m == 12 and d == 31):
            return f"{y}"
        elif d == 1:
            return f"{y}-{m:02d}"
        else:
            return f"{y}-{m:02d}-{d:02d}"
    
    # 如果是numpy datetime64
    if isinstance(date_val, np.datetime64):
        try:
            ts = pd.Timestamp(date_val)
            return smart_date_fmt(ts)  # 递归调用处理Timestamp
        except:
            return str(date_val)
    
    # 如果是字符串
    if isinstance(date_val, str):
        if len(date_val) == 4 and date_val.isdigit():
            return date_val
        elif date_val.count('-') == 2:
            parts = date_val.split('-')
            if len(parts) == 3 and ((parts[1] == "01" and parts[2] == "01") or (parts[1] == "12" and parts[2] == "31")):
                return parts[0]
        return date_val
    
    # 尝试转换为pandas Timestamp
    try:
        ts = pd.Timestamp(date_val)
        return smart_date_fmt(ts)
    except:
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
