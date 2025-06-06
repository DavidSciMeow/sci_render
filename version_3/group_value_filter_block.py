from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QPushButton, QScrollArea, QCheckBox, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt
from utils import infer_col_type
from numeric_filter_dialog import NumericFilterDialog

class GroupValueFilterBlock(QWidget):
    changed = pyqtSignal(str, object)
    def __init__(self, group_field, group_values, col_type, preselect=None, parent=None):
        super().__init__(parent)
        self.group_field = group_field
        self.vars = {}
        self.group_values = group_values
        self.col_type = col_type
        self.advanced_filter = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        label = QLabel(f"{group_field}分组值:")
        label.setFixedHeight(20)
        layout.addWidget(label)
        if col_type == "numeric":
            adv_layout = QHBoxLayout()
            adv_label = QLabel("高级筛选:")
            adv_label.setFixedHeight(24)
            adv_layout.addWidget(adv_label)
            self.adv_btn = QPushButton("设置")
            self.adv_btn.setFixedHeight(24)
            self.adv_btn.clicked.connect(self.open_advanced_filter)
            adv_layout.addWidget(self.adv_btn)
            adv_layout.addStretch(1)
            layout.addLayout(adv_layout)
            self.adv_desc_label = QLabel()
            self.adv_desc_label.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(self.adv_desc_label)
        else:
            btn_layout = QHBoxLayout()
            self.all_btn = QPushButton("全选")
            self.all_btn.setFixedHeight(24)
            self.all_btn.clicked.connect(self.select_all)
            btn_layout.addWidget(self.all_btn)
            self.none_btn = QPushButton("全不选")
            self.none_btn.setFixedHeight(24)
            self.none_btn.clicked.connect(self.select_none)
            btn_layout.addWidget(self.none_btn)
            layout.addLayout(btn_layout)
            self.cb_area = QWidget()
            self.cb_layout = QVBoxLayout(self.cb_area)
            self.cb_layout.setContentsMargins(0, 0, 0, 0)
            self.cb_layout.setSpacing(0)
            max_text_width = 0
            font_metrics = self.fontMetrics()
            for v in group_values:
                var = QCheckBox(str(v))
                var.setChecked(True if (preselect is None or v in preselect) else False)
                var.stateChanged.connect(self._changed)
                self.cb_layout.addWidget(var, alignment=Qt.AlignmentFlag.AlignLeft)
                self.vars[v] = var
                max_text_width = max(max_text_width, font_metrics.horizontalAdvance(str(v)))
            self.cb_layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.cb_area)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            min_width = max(120, max_text_width + 48)
            self.setMinimumWidth(min_width)
            self.setMaximumWidth(min_width + 30)
            layout.addWidget(scroll)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

    def open_advanced_filter(self):
        min_val = min(self.group_values) if self.group_values else 0
        max_val = max(self.group_values) if self.group_values else 0
        init_filters = self.advanced_filter.get("filters") if self.advanced_filter else None
        dialog = NumericFilterDialog(min_val, max_val, self, init_filters)
        if dialog.exec() == dialog.DialogCode.Accepted:
            result = dialog.get_filters()
            desc = self._build_advanced_desc(result)
            self.advanced_filter = {"filters": result, "desc": desc}
            self.update_advanced_desc()
            self.changed.emit(self.group_field, self.get_selected())

    def _build_advanced_desc(self, filters):
        descs = []
        for f in filters:
            mode = f.get("mode", "区间")
            lower = f.get("lower", -1e12)
            upper = f.get("upper", 1e12)
            logic = f.get("logic", "且")
            if mode == "区间":
                s = f"[{lower}, {upper}]"
            elif mode == ">=":
                s = f">= {lower}"
            elif mode == "<=" :
                s = f"<= {upper}"
            elif mode == "=":
                s = f"= {lower}"
            elif mode == "!=":
                s = f"!= {lower}"
            else:
                s = f"{mode} {lower},{upper}"
            if descs:
                s = logic + s
            descs.append(s)
        return " ".join(descs)

    def update_advanced_desc(self):
        if self.advanced_filter:
            desc = self.advanced_filter.get('desc', '已设置高级筛选')
            self.adv_desc_label.setText(f"高级筛选: {desc}")
        else:
            self.adv_desc_label.setText("")

    def _changed(self):
        if self.col_type == "numeric":
            if self.advanced_filter:
                self.changed.emit(self.group_field, self.get_selected())
            else:
                mode = self.mode_cb.currentText()
                lower = self.lower_sb.value()
                upper = self.upper_sb.value()
                self.changed.emit(self.group_field, {"mode": mode, "lower": lower, "upper": upper})
        else:
            selected = [k for k, v in self.vars.items() if v.isChecked()]
            self.changed.emit(self.group_field, selected)

    def select_all(self):
        if self.col_type != "numeric":
            for v in self.vars.values():
                v.blockSignals(True)
                v.setChecked(True)
                v.blockSignals(False)

    def select_none(self):
        if self.col_type != "numeric":
            for v in self.vars.values():
                v.blockSignals(True)
                v.setChecked(False)
                v.blockSignals(False)

    def get_selected(self):
        if self.col_type == "numeric":
            if self.advanced_filter:
                return {"advanced": self.advanced_filter}
            return None  # 没有筛选条件时返回None
        else:
            return [k for k, v in self.vars.items() if v.isChecked()]

    def set_selected(self, selected_list):
        if self.col_type == "numeric":
            if isinstance(selected_list, dict):
                if "advanced" in selected_list:
                    self.advanced_filter = selected_list["advanced"]
                    self.update_advanced_desc()
                else:
                    self.advanced_filter = None
                    mode = selected_list.get("mode", "区间")
                    lower = selected_list.get("lower", None)
                    upper = selected_list.get("upper", None)
                    self.mode_cb.setCurrentText(mode)
                    if lower is not None:
                        self.lower_sb.setValue(lower)
                    if upper is not None:
                        self.upper_sb.setValue(upper)
        else:
            for k, v in self.vars.items():
                v.blockSignals(True)
                v.setChecked(k in selected_list)
                v.blockSignals(False)
