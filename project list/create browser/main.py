from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

import sys

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置窗口标题
        self.setWindowTitle('My Browser')
        # 设置窗口图标
        self.setWindowIcon(QIcon('pic/web.png'))
        self.show()

        # 添加 URL 地址栏
        self.urlbar = QLineEdit()
        # 让地址栏能响应回车按键信号
        self.urlbar.returnPressed.connect(self.navigate_to_url)

        # 设置浏览器
        self.browser = QWebEngineView()
        url = 'https://www.baidu.com/'

        # 指定打开界面的 URL
        self.browser.setUrl(QUrl(url))
         # 添加浏览器到窗口中
        self.setCentralWidget(self.browser)

        # 添加导航栏
        navigation_bar = QToolBar("Navigation")
        navigation_bar.setIconSize(QSize(20,20))
        self.addToolBar(navigation_bar)

        # 添加前进、后退、停止加载和刷新的按钮
        back_button = QAction(QIcon('pic/back.jpeg'), 'Back', self)
        next_button = QAction(QIcon('pic/forward.png'), 'Forward', self)
        stop_button = QAction(QIcon('pic/stop.jpeg'), 'stop', self)
        reload_button = QAction(QIcon('pic/renew.png'), 'reload', self)

        back_button.triggered.connect(self.browser.back)
        next_button.triggered.connect(self.browser.forward)
        stop_button.triggered.connect(self.browser.stop)
        reload_button.triggered.connect(self.browser.reload)

        # 将按钮添加到导航栏上
        navigation_bar.addAction(back_button)
        navigation_bar.addAction(next_button)
        navigation_bar.addAction(stop_button)
        navigation_bar.addAction(reload_button)



        navigation_bar.addSeparator()
        navigation_bar.addWidget(self.urlbar)

        self.browser.urlChanged.connect(self.renew_urlbar)

    def renew_urlbar(self, q, browser=None):
        # 如果不是当前窗口所展示的网页则不更新 URL
        # if browser != self.tabs.currentWidget():
        #     return
        # 将当前网页的链接更新到地址栏
        self.urlbar.setText(q.toString())
        self.urlbar.setCursorPosition(0)

    # 响应回车按钮，将浏览器当前访问的 URL 设置为用户输入的 URL
    def navigate_to_url(self):

        q = QUrl(self.urlbar.text())
        if q.scheme() == '':
            q.setScheme('http')
        self.browser.setUrl(q)


# 创建应用
app = QApplication(sys.argv)
# 创建主窗口
window = MainWindow()
# 显示窗口
window.show()
# 运行应用，并监听事件
app.exec_()