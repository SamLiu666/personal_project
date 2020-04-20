from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 设置窗口标题
        self.setWindowTitle('My First Browser')
        label = QLabel("Welcome to Browser")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

        #  创建工具栏
        tb = QToolBar("Tool Bar")
        tb.setIconSize(QSize(32,32))
        self.addToolBar(tb)

        button_action = QAction(QIcon('share.jpg'), 'Menu button', self)
        # 设置状态栏提示
        button_action.setStatusTip('This is menu button')
        button_action.triggered.connect(self.onButtonClick)
        button_action.setCheckable(True)
        # 添加到工具栏
        tb.addAction(button_action)
        # 为主窗口设置状态栏
        self.setStatusBar(QStatusBar(self))

        # 添加新的菜单选项
        button_action2 = QAction('C++', self)
        button_action3 = QAction('Python', self)
        button_action2.setCheckable(True)
        button_action3.setCheckable(True)
        button_action2.triggered.connect(self.onButtonClick)
        button_action3.triggered.connect(self.onButtonClick)

        # 添加菜单栏
        mb = self.menuBar()
        # 禁用原生的菜单栏
        mb.setNativeMenuBar(False)
        # 添加 “文件” 菜单
        file_menu = mb.addMenu('&File')
        # 为文件菜单添加动作
        file_menu.addAction(button_action)
        # 为菜单选项添加分隔符
        file_menu.addSeparator()

        # 添加二级菜单
        build_system_menu = file_menu.addMenu('&Build System')
        build_system_menu.addAction(button_action2)
        build_system_menu.addSeparator()
        build_system_menu.addAction(button_action3)

        # # 添加布局
        # layout = QHBoxLayout()
        #
        # # 创建按钮
        # for i in range(5):
        #     button = QPushButton(str(i))
        #     # 将按钮按压信号与自定义函数关联
        #     button.pressed.connect(lambda x=i: self._my_func(x))
        #     # 将按钮添加到布局中
        #     layout.addWidget(button)
        #
        # # 创建部件
        # widget = QWidget()
        # # 将布局添加到部件
        # widget.setLayout(layout)
        # # 将部件添加到主窗口上
        # self.setCentralWidget(widget)


    # 自定义的信号处理函数
    def _my_func(self, n):
        print('click button %s' % n)

    def onButtonClick(self, s):
        print(s)


if __name__ == '__main__':

    # 创建应用实例，通过 sys.argv 传入命令行参数
    app = QApplication(sys.argv)
    # 创建窗口实例
    window = MainWindow()
    # 显示窗口
    window.show()
    # 执行应用，进入事件循环
    app.exec_()