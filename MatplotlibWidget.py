import sys
import random
import matplotlib
import seaborn as sns
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class MyMplCanvas(FigureCanvas):
    """FigureCanvas的最终的父类其实是QWidget。"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        # 配置中文显示
        plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        self.fig = Figure(figsize=(width, height), dpi=dpi)  # 新建一个figure
        self.axes = self.fig.add_subplot(111)  # 建立一个子图，如果要建立复合图，可以在这里修改

        #self.axes.hold(False)  # 每次绘图的时候不保留上一次绘图的结果

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        '''定义FigureCanvas的尺寸策略，这部分的意思是设置FigureCanvas，使之尽可能的向外填充空间。'''
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    '''绘制静态图，可以在这里定义自己的绘图逻辑'''

    def start_plot(self, title, xlabel, ylabel, x, y):
        self.axes.cla()
        self.fig.suptitle(title)
        self.axes.scatter(x, y)
        self.axes.set_ylabel('Y轴:' + ylabel)
        self.axes.set_xlabel('X轴:' + xlabel)
        self.axes.grid(True)
        self.draw()
    
    def Oxygen_plot(self, title, xlabel, ylabel, x, predictions, real):
        self.axes.cla()
        self.fig.suptitle(title)
        self.axes.plot(x, predictions, 'b-', label='预测值')
        self.axes.plot(x, real, 'g-', label='实际值')
        self.axes.set_ylabel('Y轴:' + ylabel)
        self.axes.set_xlabel('X轴:' + xlabel)
        self.axes.grid(True)
        self.draw()

    def draw_heatmap(self, norm_data):
        self.axes.cla()
        corrmat = norm_data.corr()
        sns.heatmap(corrmat, square=True)
        plt.show()
        self.draw()

    def EfficiencyImprove_plot(self, title, xlabel, ylabel, x, new, old):
        self.fig.suptitle(title)
        self.axes.scatter(x, old, c='g')
        self.axes.scatter(x, new, c='b')
        self.axes.set_ylabel('Y轴:' + ylabel)
        self.axes.set_xlabel('X轴:' + xlabel)
        self.axes.grid(True)
        self.draw()

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=5, height=4, dpi=100)
        #self.mpl.start_plot() # 如果你想要初始化的时候就呈现静态图，请把这行注释去掉
        self.mpl_ntb = NavigationToolbar(self.mpl, self)  # 添加完整的 toolbar

        self.layout.addWidget(self.mpl)
        self.layout.addWidget(self.mpl_ntb)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MatplotlibWidget()
    #ui.mpl.start_plot()  # 测试静态图效果
    # ui.mpl.start_dynamic_plot() # 测试动态图效果
    ui.show()
    sys.exit(app.exec_()) 