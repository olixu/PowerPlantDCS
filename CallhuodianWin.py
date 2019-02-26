# -*- coding:utf-8 -*-
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
from huodianWin import Ui_Form
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import datetime
import pandas as pd
import logging
import logging.config
from MatplotlibWidget import *
from sklearn import preprocessing
logging.config.fileConfig("database/log_config.config")


# 数据预处理标签下导入数据的线程
class InputDate_Thread(QThread):
    # 定义信号：str, str, pd.DataFrame
    sinOut = pyqtSignal(str, pd.DataFrame)

    def __init__(self):
        super(InputDate_Thread, self).__init__()
        self.filename = None

    def get_filename(self, filename):
        self.filename = filename
        print(filename)

    def run(self):
        starttime = datetime.datetime.now()
        plant_data = pd.read_excel(self.filename[0])
        plant_data.drop([0], inplace=True)
        plant_data.drop([1], inplace=True)
        endtime = datetime.datetime.now()
        time_cost = endtime-starttime
        self.sinOut.emit(str(time_cost), plant_data)

# 主窗口的MainWindow类


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 日志初始化
        self.logger = logging.getLogger(__name__)
        # 各个线程初始化
        # 1. 连接信号和槽函数
        self.DataMiningthread = InputDate_Thread()
        self.DataMiningthread.sinOut.connect(
            self.DataMiningLabel_Change_Status)

        # 定义一些变量

        self.df_rows = 8
        self.df_cols = 11
        self.plant_data = None
        # 初始化
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 输出启动日志
        self.logger.info("启动了程序")
        self.ui.LogTextEdit.insertPlainText("启动了程序")

    # 数据预处理标签下的槽函数：导入数据->OpenFile()
    def OpenFile(self):
        self.filename = QFileDialog.getOpenFileName(
            self, 'open file', '/media/boss/大学资料/大学4年学习资料/大四下/毕业设计所有资料/嘉兴数据/完整版/20160612-0624')
        self.ui.DataMiningLabel.setText("正在导入，文件较大，请等待")
        self.logger.info("正在导入，文件较大，请等待")
        self.DataMiningthread.get_filename(self.filename)
        self.DataMiningthread.start()

    # 数据预处理标签下的Label提示框显示信息变化
    def DataMiningLabel_Change_Status(self, time_cost, plant_data):
        self.plant_data = plant_data
        self.ui.DataMiningLabel.setText("导入完成，共耗费："+str(time_cost))
        self.logger.info("导入完成，共耗费："+str(time_cost))

    # 数据预处理标签下的槽函数：数据预处理->Calculate()
    def Calculate(self):
        # plant_data.drop([1], inplace=True)
        raw_给煤量 = ['给煤机A给煤量', '给煤机B给煤量', '给煤机C给煤量',
                   '给煤机D给煤量', '给煤机E给煤量', '给煤机F给煤量']
        raw_主蒸汽压力 = ['主蒸汽压力']
        raw_炉膛出口烟温度 = ['炉膛烟温探针B']
        raw_主蒸汽温度 = ['主蒸汽温度1']
        raw_机组负荷 = ['负荷']
        raw_给水温度 = ['省煤器进口给水温度']
        raw_给水流量 = ['给水流量']
        raw_一次风量 = ['总一次风量']
        # raw_一次风温 = ['']
        raw_二次风量 = ['A侧二次风量', 'B侧二次风量']
        raw_二次风温 = ['二次风温度']
        raw_含氧量 = ['省煤器A侧出口烟气氧量分析1', '省煤器 A侧出口烟气氧量分析2', '省煤器A侧出口烟气氧量分析3',
                   '省煤器B侧出口烟气氧量分析1', '省煤器B侧出口烟气氧量分析2', '省煤器B侧出口烟气氧量分析3']
        # raw_含氧量 = ['省煤器A侧出口烟气氧量分析3']
        给煤量 = self.plant_data[raw_给煤量].sum(1)
        主蒸汽压力 = self.plant_data[raw_主蒸汽压力]
        炉膛出口烟温度 = self.plant_data[raw_炉膛出口烟温度]
        主蒸汽温度 = self.plant_data[raw_主蒸汽温度]
        机组负荷 = self.plant_data[raw_机组负荷]
        给水温度 = self.plant_data[raw_给水温度]
        给水流量 = self.plant_data[raw_给水流量]
        一次风量 = self.plant_data[raw_一次风量]
        二次风量 = self.plant_data[raw_二次风量].mean(1)
        二次风温 = self.plant_data[raw_二次风温]
        含氧量 = self.plant_data[raw_含氧量].mean(1)
        fields = [给煤量, 主蒸汽压力, 炉膛出口烟温度, 主蒸汽温度, 机组负荷,
                  给水温度, 给水流量, 一次风量, 二次风量, 二次风温, 含氧量]
        outcome = pd.concat(fields, axis=1)
        outcome.columns = ['给煤量', '主蒸汽压力', '炉膛出口烟温度', '主蒸汽温度',
                           '机组负荷', '给水温度', '给水流量', '一次风量', '二次风量', '二次风温', '含氧量']
        # outcome['机组负荷'].describe()
        # outcome.dtypes
        outcome = outcome.apply(pd.to_numeric)
        self.data_pre_handle = outcome
        ruanceliang_data = outcome.describe()

        self.ui.DataMiningTableWidget.setRowCount(self.df_rows)
        self.ui.DataMiningTableWidget.setColumnCount(self.df_cols)
        self.ui.DataMiningTableWidget.setHorizontalHeaderLabels(
            ['给煤量', '主蒸汽压力', '炉膛出口烟温度', '主蒸汽温度', '机组负荷', '给水温度', '给水流量', '一次风量', '二次风量', '二次风温', '含氧量'])
        self.ui.DataMiningTableWidget.setVerticalHeaderLabels(
            ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        for i in range(self.df_rows):
            for j in range(self.df_cols):
                x = '{:.3f}'.format(ruanceliang_data.iloc[i, j])
                self.ui.DataMiningTableWidget.setItem(
                    i, j, QTableWidgetItem(x))
        self.ui.DataMiningTableWidget.resizeColumnsToContents()
        self.ui.DataMiningTableWidget.resizeRowsToContents()

    # 数据可视化标签下的槽函数：画图->DataVisualPlot(self, title, xlabel, ylabel, x, y)
    def DataVisualPlot(self):
        # DataVisulWebEngine
        # self.ui.DataVisualWebEngine.load(QUrl.fromLocalFile('home/boss/Desktop/pyqttest/PyQt5-master/Chapter09/if_hs300_bais.html'))
        self.ui.DataVisualWidget.setVisible(True)
        self.ui.DataVisualWidget.mpl.start_plot(self.ui.XComboBox.currentText()+'-'+self.ui.YComboBox.currentText(),
												self.ui.XComboBox.currentText(),
												self.ui.YComboBox.currentText(),
												self.data_pre_handle[self.ui.XComboBox.currentText()],
												self.data_pre_handle[self.ui.YComboBox.currentText()]
												)
        print("正在数据可视化画图")
        self.logger.info("画了一幅图片，参数是：")
        print(self.ui.XComboBox.currentText())
        print("输出结果完毕")
        # print(self.ui.XComboBox.currentText()+'-'+self.ui.YComboBox.currentText())
        # print(self.ui.XComboBox.currentText())
        # print(self.ui.YComboBox.currentText())
        # print(self.data_pre_handle[self.ui.XComboBox.currentText()].shape)
        # print(self.data_pre_handle[self.ui.YComboBox.currentText()].shape)

	# 数据可视化标签下的槽函数：Heatmap->HeapMapPlot()
    def HeapMapPlot(self):
        # DataVisulWebEngine
        # self.ui.DataVisualWebEngine.load(QUrl.fromLocalFile('home/boss/Desktop/pyqttest/PyQt5-master/Chapter09/if_hs300_bais.html'))
        norm_data = (self.data_pre_handle - self.data_pre_handle.min())/(self.data_pre_handle.max() - self.data_pre_handle.min())
        self.ui.DataVisualWidget.setVisible(True)
        self.ui.DataVisualWidget.mpl.draw_heatmap(norm_data)
        print("正在数据可视化画图")
        self.logger.info("画了一幅图片，参数是：")
        print(self.ui.XComboBox.currentText())
        print("输出结果完毕")

    # 烟气含氧量软测量下的槽函数：开始画图->OxygenVisualPlot()
    def OxygenVisualPlot(self):
        # self.ui.OxygenWebEngine.load(QUrl.fromLocalFile('home/boss/Desktop/pyqttest/PyQt5-master/Chapter09/if_hs300_bais.html'))
        self.ui.OxygenVisualWidget.setVisible(True)
        self.ui.OxygenVisualWidget.mpl.start_plot()
        print("正在烟气含氧量画图")
        self.logger.info("画了一幅图片，参数是：")


if __name__ == '__main__':
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec_())
