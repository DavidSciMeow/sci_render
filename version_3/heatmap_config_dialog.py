from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QComboBox, QDialogButtonBox
from PyQt6.QtCore import Qt

class HeatmapConfigDialog(QDialog):
    def __init__(self, columns, parent=None, preselect_cols=None):
        super().__init__(parent)
        self.setWindowTitle("热力图参数设置")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请选择要分析的数值列："))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in columns:
            self.col_list.addItem(col)
        self.col_list.setMinimumWidth(200)
        self.col_list.setSizeAdjustPolicy(QListWidget.SizeAdjustPolicy.AdjustToContents)
        self.col_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.col_list.setWordWrap(True)
        self.col_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.col_list.setUniformItemSizes(False)
        self.col_list.setWrapping(False)
        self.col_list.setSizePolicy(self.col_list.sizePolicy().horizontalPolicy(), self.col_list.sizePolicy().verticalPolicy())
        max_col_width = max([self.col_list.fontMetrics().horizontalAdvance(col) for col in columns]) if columns else 100
        self.col_list.setMinimumWidth(max(200, max_col_width + 40))
        layout.addWidget(self.col_list)
        layout.addWidget(QLabel("热力图展示方式："))
        self.triangle_cb = QComboBox()
        self.triangle_cb.addItems(["全部", "下三角", "上三角"])
        self.triangle_cb.setCurrentText("下三角")
        layout.addWidget(self.triangle_cb)
        layout.addWidget(QLabel("热力图生成模式："))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["中文（有相关性分析）", "不带数字（只有相关性分析）", "仅有数字"])
        layout.addWidget(self.mode_cb)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        # 预选列
        if preselect_cols:
            for i in range(self.col_list.count()):
                item = self.col_list.item(i)
                if item.text() in preselect_cols:
                    item.setSelected(True)

    def get_config(self):
        cols = [item.text() for item in self.col_list.selectedItems()]
        triangle = self.triangle_cb.currentText()
        mode = self.mode_cb.currentText()
        return cols, triangle, mode

__all__ = ["HeatmapConfigDialog"]
