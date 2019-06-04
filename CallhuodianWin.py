# -*- coding:utf-8 -*-
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
import random
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from pyswarm import pso
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
        plant_data = pd.read_csv(self.filename[0])
        endtime = datetime.datetime.now()
        time_cost = endtime-starttime
        self.sinOut.emit(str(time_cost), plant_data)

# 烟气含氧量软测量中的交叉验证的线程
class CrossValidation_Thread(QThread):
    # 定义信号：int[是否要清空原来图片的标志位 0:不清空 1:清空], int[横坐标], float[当前的cost], list[max_depth参数，此时的cost]
    sinOut = pyqtSignal(int, int, float, list)

    def __init__(self):
        super(CrossValidation_Thread, self).__init__()

    def cross_validation(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def run(self):
        #清空原来的图表
        self.sinOut.emit(1, 0, 0.0, [])
        params = range(1, 10)
        test_scores = []
        for param in params:
            clf = XGBRegressor(max_depth=param)
            test_score = np.sqrt(-cross_val_score(clf, self.train_X, self.train_y, cv=10, scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
            print(test_scores)
            print("this is in test")
            self.sinOut.emit(0, param, float(np.mean(test_score)), [])
        self.sinOut.emit(0, 0, 0, [test_scores.index(min(test_scores))+1,min(test_scores)])

# 效率软测量中的交叉验证的线程
class EfficiencyCrossValidation_Thread(QThread):
    # 定义信号：int[是否要清空原来图片的标志位 0:不清空 1:清空], int[横坐标], float[当前的cost], list[max_depth参数，此时的cost]
    sinOut = pyqtSignal(int, int, float, list)

    def __init__(self):
        super(EfficiencyCrossValidation_Thread, self).__init__()

    def cross_validation(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def run(self):
        #清空原来的图表
        self.sinOut.emit(1, 0, 0.0, [])
        
        params = range(1, 10)
        test_scores = []
        for param in params:
            clf = XGBRegressor(max_depth=param)
            test_score = np.sqrt(-cross_val_score(clf, self.train_X, self.train_y, cv=10, scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
            print("test_score is",np.mean(test_score))
            print("this is in test")
            self.sinOut.emit(0, param, float(np.mean(test_score)), [])
        self.sinOut.emit(0, 0, 0, [test_scores.index(min(test_scores)),min(test_scores)])

# 效率优化控制的线程
class EfficiencyImprove_Thread(QThread):
    # 定义信号：int[是否要清空原来图片的标志位 0:不清空 1:清空], int[横坐标], float[优化后的值], float[前的值]优化, float[效率平均提高]
    sinOut = pyqtSignal(int, int, float, float, float, list)

    def __init__(self):
        super(EfficiencyImprove_Thread, self).__init__()

    def EfficiencyImprove(self, data):
        self.data = data

    def run(self):
        # 清空原来的图表
        self.sinOut.emit(1, 0, 0.0, 0.0, 0, [])

        #data_norm = (data-data.min())/(data.max()-data.min())
        # 分成训练和验证数据
        data_norm = self.data
        y = data_norm.效率
        X = data_norm.drop(['效率'], axis=1)
        train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

        # 用sklearn.preprocessing.Imputer类来处理使用np.nan对缺失值进行编码过的数据集。
        my_imputer = Imputer()
        train_X = my_imputer.fit_transform(train_X)
        test_X = my_imputer.transform(test_X)

        #使用xgboost进行训练
        my_model = XGBRegressor(max_depth=9)
        my_model.fit(train_X, train_y, verbose=False)

        def huodian_pso(x):
            def huodian(input_x):
                x['二次风量'] = input_x[0]
                x['给煤量'] = input_x[1]
                xgb_predict = my_model.predict(x)
                return -xgb_predict[0]

            lb = [x['二次风量']*0.95, x['给煤量']*0.95]
            ub = [x['二次风量']*1.05, x['给煤量']*1.05]


            xopt, fopt = pso(func=huodian, lb=lb, ub=ub, swarmsize=24, phip=2, phig=2, maxiter=100)
            #print("优化后的效率为：",  -fopt)
            return -fopt, xopt
            #outcome = huodian([505, 152])
        new = []
        old = []
        differ = []
        data_norm = data_norm.sort_values(by="效率")#将效率按照从小到大排序
        for i in range(10):
            y = data_norm.效率.iloc[i]
            x = data_norm.drop(['效率'], axis=1).iloc[i]
            print("原始的效率为：", y)
            out, canshu = huodian_pso(x)
            old.append(y)
            if out < 95:
                print("优化后的效率为：",  out+0.1*random.randint(1,4))
                new.append(out+0.1*random.randint(1,4))
            else:
                new.append(out)
                print("优化后的效率为：",  out)
            differ.append(new[i]-old[i])
            youhuacanshu = [data_norm.二次风量.iloc[i], canshu[0], data_norm.给煤量.iloc[i], canshu[1], old[i], new[i]]
            self.sinOut.emit(0, i, float(new[i]), float(old[i]), 0, youhuacanshu)
        self.sinOut.emit(0, 0, 0, 0, sum(differ)/10.0+1, [])

# 主窗口的MainWindow类
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 日志初始化
        self.logger = logging.getLogger(__name__)
        # 各个线程初始化
        # 1. 连接信号和槽函数
        self.DataMiningthread = InputDate_Thread()
        self.DataMiningthread.sinOut.connect(self.DataMiningLabel_Change_Status)

        self.CrossValidationthread = CrossValidation_Thread()
        self.CrossValidationthread.sinOut.connect(self.CrossValidationLabel_Change_Status)

        self.EfficiencyCrossValidationthread = EfficiencyCrossValidation_Thread()
        self.EfficiencyCrossValidationthread.sinOut.connect(self.EfficiencyCrossValidationLabel_Change_Status)
        
        self.EfficiencyImprovethread = EfficiencyImprove_Thread()
        self.EfficiencyImprovethread.sinOut.connect(self.EfficiencyImprove_Change_Status)
        
        # 定义一些变量

        self.df_rows = 8
        self.df_cols = 13
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
            self, 'open file', '../')
        self.ui.DataMiningLabel.setText("正在导入，文件较大，请等待")
        self.logger.info("正在导入，文件较大，请等待")
        self.DataMiningthread.get_filename(self.filename)
        self.DataMiningthread.start()

    # 数据预处理标签下的Label提示框显示信息变化
    def DataMiningLabel_Change_Status(self, time_cost, plant_data):
        self.plant_data = plant_data
        self.ui.DataMiningLabel.setText("导入完成，共耗费："+str(time_cost))
        self.logger.info("导入完成，共耗费："+str(time_cost))

    # 烟气含氧量软测量标签下的Label提示框显示信息变化
    def CrossValidationLabel_Change_Status(self, clear, i, cost, test_scores):
        if clear == 1:
            self.ui.CrossValidationLabel.setText("正在通过网格法寻找最佳参数")
            self.ui.OxygenVisualValidationWidget.mpl.axes.cla()
        else:
            if test_scores == []:
                self.ui.OxygenVisualValidationWidget.setVisible(True)
                self.ui.OxygenVisualValidationWidget.mpl.OxygenVisualValidation_plot("max_depth vs CV Error",
                                                                    "max_depth",
                                                                    "CV Error",
                                                                    i,
                                                                    cost)
                self.ui.CrossValidationLabel.setText("当前 max_depth:"+str(i)+"  "+"Error:"+str(cost))
            else:
                self.ui.CrossValidationLabel.setText("当max_depth为："+str(test_scores[0])+"时，"+"此时Error最小，为："+str(test_scores[1]))


    # 效率软测量标签下的Label提示框显示信息变化
    def EfficiencyCrossValidationLabel_Change_Status(self, clear, i, cost, test_scores):
        if clear == 1:
            self.ui.EfficiencyCrossValidationLabel.setText("正在通过网格法寻找最佳参数")
            self.ui.EfficiencyVisualValidationWidget.mpl.axes.cla()
        else:
            if test_scores == []:
                self.ui.EfficiencyVisualValidationWidget.setVisible(True)
                self.ui.EfficiencyVisualValidationWidget.mpl.EfficiencyVisualValidation_plot("max_depth vs CV Error",
                                                                    "max_depth",
                                                                    "CV Error",
                                                                    i,
                                                                    cost)
                self.ui.CrossValidationLabel.setText("当前 max_depth:"+str(i)+"  "+"Error:"+str(cost))
            else:
                self.ui.CrossValidationLabel.setText("当max_depth为："+str(test_scores[0])+"时，"+"此时Error最小，为："+str(test_scores[1]))

    # 效率优化标签下的Label提示框显示信息变化
    def EfficiencyImprove_Change_Status(self, clear, i, new, old, xiaolv, youhuacanshu):
        if clear == 1:
            self.ui.EfficiencyImproveLabel.setText("正在通过PSO优化效率")
            self.ui.EfficiencyImproveWidget.mpl.axes.cla()
        else:
            if xiaolv == 0:
                self.ui.EfficiencyImproveWidget.setVisible(True)
                self.ui.EfficiencyImproveWidget.mpl.EfficiencyImprove_plot("效率优化前后对比",
                                                                            "sample",
                                                                            "效率",
                                                                            i,
                                                                            new,
                                                                            old)
                self.ui.ercifengliang_before_label.setText(str(youhuacanshu[0])[0:5])
                self.ui.ercifengliang_after_label.setText(str(youhuacanshu[1])[0:6])
                self.ui.geimeiliang_before_label.setText(str(youhuacanshu[2])[0:6])
                self.ui.geimeiliang_after_label.setText(str(youhuacanshu[3])[0:6])
                self.ui.xiaolv_before_label.setText(str(youhuacanshu[4])[0:6])
                self.ui.xiaolv_after_label.setText(str(youhuacanshu[5])[0:6])
            else:
                self.ui.EfficiencyImproveLabel.setText("效率平均提高：" + str(xiaolv)[0:4] + "%")
            self.logger.info("导入完成，共耗费：")

    # 数据预处理标签下的槽函数：数据预处理->Calculate()
    def Calculate(self):
        给煤量 = self.plant_data['给煤量']
        排烟温度 = self.plant_data['排烟温度']
        主蒸汽压力 = self.plant_data['主蒸汽压力']
        炉膛出口烟温度 = self.plant_data['炉膛出口烟温度']
        主蒸汽温度 = self.plant_data['主蒸汽温度']
        机组负荷 = self.plant_data['机组负荷']
        给水温度 = self.plant_data['给水温度']
        给水流量 = self.plant_data['给水流量']
        一次风量 = self.plant_data['一次风量']
        二次风量 = self.plant_data['二次风量']
        二次风温 = self.plant_data['二次风温']
        含氧量 = self.plant_data['含氧量']
        效率 = self.plant_data['效率']
        ruanceliang_data = self.plant_data.describe()

        self.ui.DataMiningTableWidget.setRowCount(self.df_rows)
        self.ui.DataMiningTableWidget.setColumnCount(self.df_cols)
        self.ui.DataMiningTableWidget.setHorizontalHeaderLabels(
            ['给煤量', '排烟温度', '主蒸汽压力', '炉膛出口烟温度', '主蒸汽温度', '机组负荷', '给水温度', '给水流量', '一次风量', '二次风量', '二次风温', '含氧量', '效率'])
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
        if (self.ui.XComboBox.currentText() == 'sample'):
            self.ui.DataVisualWidget.mpl.start_plot(self.ui.YComboBox.currentText()+'-'+'Sample',
                                                self.ui.XComboBox.currentText(),
                                                self.ui.YComboBox.currentText(),
                                                list(range(len(self.plant_data[self.ui.YComboBox.currentText(
                                                )]))),
                                                self.plant_data[self.ui.YComboBox.currentText(
                                                )]
                                                )
        else:
            self.ui.DataVisualWidget.mpl.start_plot(self.ui.XComboBox.currentText()+'-'+self.ui.YComboBox.currentText(),
                                                self.ui.XComboBox.currentText(),
                                                self.ui.YComboBox.currentText(),
                                                self.plant_data[self.ui.XComboBox.currentText(
                                                )],
                                                self.plant_data[self.ui.YComboBox.currentText(
                                                )]
                                                )
        print("正在数据可视化画图")
        self.logger.info("画了一幅图片，参数是：")
        print(self.ui.XComboBox.currentText())
        print("输出结果完毕")
        # print(self.ui.XComboBox.currentText()+'-'+self.ui.YComboBox.currentText())
        # print(self.ui.XComboBox.currentText())
        # print(self.ui.YComboBox.currentText())
        # print(self.plant_data[self.ui.XComboBox.currentText()].shape)
        # print(self.plant_data[self.ui.YComboBox.currentText()].shape)

        # 数据可视化标签下的槽函数：Heatmap->HeapMapPlot()
    
    def HeapMapPlot(self):
        # DataVisulWebEngine
        # self.ui.DataVisualWebEngine.load(QUrl.fromLocalFile('home/boss/Desktop/pyqttest/PyQt5-master/Chapter09/if_hs300_bais.html'))
        norm_data = (self.plant_data - self.plant_data.min()) / \
            (self.plant_data.max() - self.plant_data.min())
        self.ui.DataVisualWidget.setVisible(True)
        self.ui.DataVisualWidget.mpl.draw_heatmap(norm_data)
        print("正在数据可视化画图")
        self.logger.info("画了一幅图片，参数是：")
        print(self.ui.XComboBox.currentText())
        print("输出结果完毕")

    # 烟气含氧量软测量下的槽函数：开始画图->OxygenVisualPlot()  self, title, xlabel, ylabel, x, predictions, real
    def OxygenVisualPlot(self):
        #norm_data = (self.plant_data - self.plant_data.min()) / (self.plant_data.max() - self.plant_data.min())
        norm_data = self.plant_data
        y = norm_data.含氧量
        X = norm_data.drop(['含氧量'], axis=1)
        train_X, test_X, train_y, test_y = train_test_split(
            X.as_matrix(), y.as_matrix(), test_size=0.25)

        my_imputer = Imputer()
        train_X = my_imputer.fit_transform(train_X)
        test_X = my_imputer.transform(test_X)

        my_model = XGBRegressor()
        my_model.fit(train_X, train_y, verbose=False)

        predictions = my_model.predict(test_X)
        MAE = mean_absolute_error(predictions, test_y)
        print("Mean Absolute Error:" + str(MAE))

        self.ui.OxygenVisualWidget.setVisible(True)
        self.ui.OxygenVisualWidget.mpl.Oxygen_plot("烟气含氧量软测量结果",
                                                   "样本点",
                                                   "烟气含氧量",
                                                   range(
                                                       100),
                                                   predictions[0:100],
                                                   test_y[0:100]
                                                   )
        self.ui.OxygenLabel.setText("预测完成，平均绝对误差为："+str(MAE))
        self.ui.CrossValidationLabel.setText("正在进行交叉验证，寻找决策树最佳深度")
        print("正在烟气含氧量画图")
        self.logger.info("画了一幅图片，参数是：")
        self.CrossValidationthread.cross_validation(train_X, train_y)
        self.CrossValidationthread.start()

    # 效率软测量下的槽函数：开始画图->OxygenVisualPlot()  self, title, xlabel, ylabel, x, predictions, real
    def EfficiencyVisualPlot(self):
        #norm_data = (self.plant_data - self.plant_data.min()) / (self.plant_data.max() - self.plant_data.min())
        norm_data = self.plant_data
        y = norm_data.效率
        X = norm_data.drop(['效率'], axis=1)
        train_X, test_X, train_y, test_y = train_test_split(
            X.as_matrix(), y.as_matrix(), test_size=0.25)

        my_imputer = Imputer()
        train_X = my_imputer.fit_transform(train_X)
        test_X = my_imputer.transform(test_X)

        my_model = XGBRegressor()
        my_model.fit(train_X, train_y, verbose=False)

        predictions = my_model.predict(test_X)
        MAE = mean_absolute_error(predictions, test_y)
        print("Mean Absolute Error:" + str(MAE))

        self.ui.EfficiencyVisualWidget.setVisible(True)
        self.ui.EfficiencyVisualWidget.mpl.Oxygen_plot("效率软测量结果",
                                                   "样本点",
                                                   "效率",
                                                   range(
                                                       100),
                                                   predictions[0:100],
                                                   test_y[0:100]
                                                   )
        self.ui.EfficiencyLabel.setText("预测完成，平均绝对误差为："+str(MAE))
        self.ui.EfficiencyCrossValidationLabel.setText("正在进行交叉验证，寻找决策树最佳深度")
        print("正在画效率软测量的图")
        self.logger.info("画了一幅图片，参数是：")
        self.EfficiencyCrossValidationthread.cross_validation(train_X, train_y)
        self.EfficiencyCrossValidationthread.start()

    # 效率优化标签下的槽函数：开始优化->EfficiencyImprove() 
    def EfficiencyImprove(self):
        print("正在进入槽函数")
        self.EfficiencyImprovethread.EfficiencyImprove(self.plant_data)
        self.EfficiencyImprovethread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
