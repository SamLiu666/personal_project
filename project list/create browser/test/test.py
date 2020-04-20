from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置窗口标题
        self.setWindowTitle('My First App')

        # 定义布局
        layout = QVBoxLayout()

        # 展示的部件列表
        widgets = [QCheckBox,
                QComboBox,
                QDateEdit,
                QDateTimeEdit,
                QDial,
                QDoubleSpinBox,
                QFontComboBox,
                QLCDNumber,
                QLineEdit,
                QProgressBar,
                QPushButton,
                QRadioButton,
                QSlider,
                QSpinBox,
                QTimeEdit]

        # 将部件添加到列表中
        for item in widgets:
            layout.addWidget(item())

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


if __name__ == '__main__':
    # 创建应用实例，通过 sys.argv 传入命令行参数
    app = QApplication(sys.argv)
    # 创建窗口实例
    window = MainWindow()
    # 显示窗口
    window.show()
    # 执行应用，进入事件循环
    app.exec_()