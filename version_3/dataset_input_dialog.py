from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QDialogButtonBox
import os
import pandas as pd

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
