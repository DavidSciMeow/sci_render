from PyQt6.QtWidgets import QDialog, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QDoubleSpinBox, QPushButton, QDialogButtonBox

class NumericFilterDialog(QDialog):
    def __init__(self, min_val, max_val, parent=None, init_filters=None):
        super().__init__(parent)
        self.setWindowTitle("高级数值筛选")
        self.filters = []  # [{mode, lower, upper, logic}]
        layout = QVBoxLayout(self)
        self.cond_area = QWidget()
        self.cond_layout = QVBoxLayout(self.cond_area)
        self.cond_layout.setContentsMargins(0, 0, 0, 0)
        self.cond_layout.setSpacing(4)
        layout.addWidget(self.cond_area)
        self.add_btn = QPushButton("添加条件")
        self.add_btn.clicked.connect(self.add_condition)
        layout.addWidget(self.add_btn)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.min_val = min_val
        self.max_val = max_val
        if init_filters:
            for f in init_filters:
                self.add_condition(f)
        else:
            self.add_condition()

    def add_condition(self, init_val=None):
        row = QHBoxLayout()
        logic_cb = QComboBox()
        logic_cb.addItems(["且", "或"])
        if len(self.filters) == 0:
            logic_cb.setVisible(False)
        mode_cb = QComboBox()
        mode_cb.addItems(["区间", ">=", "<=", "=", "!="])
        lower_sb = QDoubleSpinBox()
        lower_sb.setDecimals(6)
        lower_sb.setMinimum(-1e12)
        lower_sb.setMaximum(1e12)
        lower_sb.setValue(self.min_val)
        upper_sb = QDoubleSpinBox()
        upper_sb.setDecimals(6)
        upper_sb.setMinimum(-1e12)
        upper_sb.setMaximum(1e12)
        upper_sb.setValue(self.max_val)
        del_btn = QPushButton("删除")
        def remove():
            for i in reversed(range(row.count())):
                w = row.itemAt(i).widget()
                if w:
                    w.setParent(None)
            self.cond_layout.removeItem(row)
            self.filters.remove(cond_dict)
        del_btn.clicked.connect(remove)
        row.addWidget(logic_cb)
        row.addWidget(mode_cb)
        row.addWidget(lower_sb)
        row.addWidget(upper_sb)
        row.addWidget(del_btn)
        self.cond_layout.addLayout(row)
        cond_dict = {"logic": logic_cb, "mode": mode_cb, "lower": lower_sb, "upper": upper_sb, "row": row}
        self.filters.append(cond_dict)
        def update_spinbox_visibility():
            mode = mode_cb.currentText()
            if mode == ">=":
                lower_sb.setEnabled(True)
                lower_sb.setVisible(True)
                upper_sb.setEnabled(False)
                upper_sb.setVisible(False)
            elif mode == "<=":
                lower_sb.setEnabled(False)
                lower_sb.setVisible(False)
                upper_sb.setEnabled(True)
                upper_sb.setVisible(True)
            elif mode in ("=", "!="):
                lower_sb.setEnabled(True)
                lower_sb.setVisible(True)
                upper_sb.setEnabled(False)
                upper_sb.setVisible(False)
            else:  # 区间
                lower_sb.setEnabled(True)
                lower_sb.setVisible(True)
                upper_sb.setEnabled(True)
                upper_sb.setVisible(True)
        mode_cb.currentIndexChanged.connect(update_spinbox_visibility)
        update_spinbox_visibility()
        if init_val:
            mode_cb.setCurrentText(init_val.get("mode", "区间"))
            lower_sb.setValue(init_val.get("lower", self.min_val))
            upper_sb.setValue(init_val.get("upper", self.max_val))
            if "logic" in init_val and len(self.filters) > 1:
                logic_cb.setCurrentText(init_val["logic"])

    def get_filters(self):
        result = []
        for i, f in enumerate(self.filters):
            d = {
                "mode": f["mode"].currentText(),
                "lower": f["lower"].value(),
                "upper": f["upper"].value(),
            }
            if i > 0:
                d["logic"] = f["logic"].currentText()
            result.append(d)
        return result
