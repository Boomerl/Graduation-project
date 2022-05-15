import imp
import sys
from PyQt5 import QtGui, QtWidgets
from matplotlib import image
import run_graphsage_cora as rg
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsScene
from demo import Ui_MainWindow
import cv2


class MyThread(QThread):
    signal = pyqtSignal(str)  # 括号里填写信号传递的参数
    signal0 = pyqtSignal(str)

    def __init__(self, a, b, c, d):
        super(MyThread, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def run(self):
        try:
            self.signal0.emit("正在训练。。。")
            eval_res = rg.run(self.a, self.b, self.c, self.d)
            self.signal.emit("训练完成！")
            self.signal.emit('Test loss: ' + str(eval_res[0]))
            self.signal.emit('Test weighted_loss: ' + str(eval_res[1]))
            self.signal.emit('Test accuracy: ' + str(eval_res[2]))

        except Exception as e:
            self.exitcode = 1
            self.exitcode = e


class MyThread2(QThread):
    signal1 = pyqtSignal(str)  # 括号里填写信号传递的参数
    signal2 = pyqtSignal(str)

    def __init__(self):
        super(MyThread2, self).__init__()

    def run(self):
        self.signal2.emit('开始分类。。。')
        rg.classify()
        self.signal1.emit('.\\pic.png')


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()  # UI类的实例化()
        self.ui.setupUi(self)
        self.band()

    def band(self):
        self.ui.pushButton.clicked.connect(self.handle_click1)
        self.ui.pushButton_2.clicked.connect(self.handle_click2)

    def handle_click1(self):
        #获取参数
        func = self.ui.lineEdit.text()
        sz = self.ui.lineEdit_2.text()
        train = self.ui.lineEdit_3.text()
        epoch = self.ui.lineEdit_4.text()

        #多线程
        self.thread = MyThread(str(func), sz.split(), int(epoch), int(train))
        self.thread.signal.connect(self.callback)
        self.thread.signal0.connect(self.begin)
        self.thread.start()  # 启动线程

    def handle_click2(self):
        # 多线程
        self.thread1 = MyThread2()
        self.thread1.signal1.connect(self.showimage)
        self.thread1.signal2.connect(self.begin)
        self.thread1.start()  # 启动线程

    def callback(self, string):
        self.ui.textBrowser.append(string)

    def begin(self, str):
        self.ui.textBrowser.clear()
        self.ui.textBrowser.append(str)

    def showimage(self, p_str):
        self.ui.textBrowser.append('分类完成!')
        path = p_str
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]
        y = img.shape[0]
        frame = QtGui.QImage(img.data, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QtGui.QPixmap(pix)))  # 图像自适应大小
        self.ui.graphicsView.show()


if __name__ == '__main__':
    app = QApplication([])  # 启动一个应用
    window = MainWindow()  # 实例化主窗口
    window.show()  # 展示主窗口
    app.exec()  # 避免程序执行到这一行后直接退出