# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'huodian.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1610, 914)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(0, 20, 1561, 921))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.groupBox.setFont(font)
        self.groupBox.setTitle("")
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.tabWidget = QtWidgets.QTabWidget(self.groupBox)
        self.tabWidget.setGeometry(QtCore.QRect(-10, 50, 1571, 871))
        self.tabWidget.setObjectName("tabWidget")
        self.DataMiningTab = QtWidgets.QWidget()
        self.DataMiningTab.setObjectName("DataMiningTab")
        self.InputBtn = QtWidgets.QPushButton(self.DataMiningTab)
        self.InputBtn.setGeometry(QtCore.QRect(20, 20, 141, 41))
        self.InputBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.InputBtn.setAutoDefault(False)
        self.InputBtn.setObjectName("InputBtn")
        self.DataMiningBtn = QtWidgets.QPushButton(self.DataMiningTab)
        self.DataMiningBtn.setGeometry(QtCore.QRect(170, 20, 181, 41))
        self.DataMiningBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.DataMiningBtn.setObjectName("DataMiningBtn")
        self.DataMiningLabel = QtWidgets.QLabel(self.DataMiningTab)
        self.DataMiningLabel.setGeometry(QtCore.QRect(390, 20, 731, 41))
        self.DataMiningLabel.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.DataMiningLabel.setText("")
        self.DataMiningLabel.setTextFormat(QtCore.Qt.RichText)
        self.DataMiningLabel.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.DataMiningLabel.setObjectName("DataMiningLabel")
        self.gridLayoutWidget = QtWidgets.QWidget(self.DataMiningTab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 80, 1531, 721))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.DataMiningTableWidget = QtWidgets.QTableWidget(self.gridLayoutWidget)
        self.DataMiningTableWidget.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.DataMiningTableWidget.setObjectName("DataMiningTableWidget")
        self.DataMiningTableWidget.setColumnCount(0)
        self.DataMiningTableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.DataMiningTableWidget, 0, 0, 1, 1)
        self.tabWidget.addTab(self.DataMiningTab, "")
        self.DataVisualTab = QtWidgets.QWidget()
        self.DataVisualTab.setObjectName("DataVisualTab")
        self.DataVisualWidget = MatplotlibWidget(self.DataVisualTab)
        self.DataVisualWidget.setGeometry(QtCore.QRect(400, 90, 1131, 600))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DataVisualWidget.sizePolicy().hasHeightForWidth())
        self.DataVisualWidget.setSizePolicy(sizePolicy)
        self.DataVisualWidget.setObjectName("DataVisualWidget")
        self.DataVisualSplitter = QtWidgets.QSplitter(self.DataVisualTab)
        self.DataVisualSplitter.setGeometry(QtCore.QRect(40, 90, 321, 601))
        self.DataVisualSplitter.setOrientation(QtCore.Qt.Vertical)
        self.DataVisualSplitter.setObjectName("DataVisualSplitter")
        self.xsplitter = QtWidgets.QSplitter(self.DataVisualSplitter)
        self.xsplitter.setOrientation(QtCore.Qt.Horizontal)
        self.xsplitter.setObjectName("xsplitter")
        self.XLabel = QtWidgets.QLabel(self.xsplitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.XLabel.sizePolicy().hasHeightForWidth())
        self.XLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.XLabel.setFont(font)
        self.XLabel.setObjectName("XLabel")
        self.XComboBox = QtWidgets.QComboBox(self.xsplitter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.XComboBox.setFont(font)
        self.XComboBox.setObjectName("XComboBox")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.XComboBox.addItem("")
        self.ysplitter = QtWidgets.QSplitter(self.DataVisualSplitter)
        self.ysplitter.setOrientation(QtCore.Qt.Horizontal)
        self.ysplitter.setObjectName("ysplitter")
        self.YLabel = QtWidgets.QLabel(self.ysplitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.YLabel.sizePolicy().hasHeightForWidth())
        self.YLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.YLabel.setFont(font)
        self.YLabel.setObjectName("YLabel")
        self.YComboBox = QtWidgets.QComboBox(self.ysplitter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.YComboBox.setFont(font)
        self.YComboBox.setObjectName("YComboBox")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.YComboBox.addItem("")
        self.DataVisualBtn = QtWidgets.QPushButton(self.DataVisualSplitter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DataVisualBtn.setFont(font)
        self.DataVisualBtn.setObjectName("DataVisualBtn")
        self.DataVisualLabel = QtWidgets.QLabel(self.DataVisualSplitter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DataVisualLabel.setFont(font)
        self.DataVisualLabel.setObjectName("DataVisualLabel")
        self.HeatMapBtn = QtWidgets.QPushButton(self.DataVisualSplitter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.HeatMapBtn.setFont(font)
        self.HeatMapBtn.setObjectName("HeatMapBtn")
        self.tabWidget.addTab(self.DataVisualTab, "")
        self.OxygenTab = QtWidgets.QWidget()
        self.OxygenTab.setObjectName("OxygenTab")
        self.OxygenVisualWidget = MatplotlibWidget(self.OxygenTab)
        self.OxygenVisualWidget.setGeometry(QtCore.QRect(120, 60, 600, 480))
        self.OxygenVisualWidget.setObjectName("OxygenVisualWidget")
        self.OxygenVisualBtn = QtWidgets.QPushButton(self.OxygenTab)
        self.OxygenVisualBtn.setGeometry(QtCore.QRect(720, 680, 111, 31))
        self.OxygenVisualBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.OxygenVisualBtn.setObjectName("OxygenVisualBtn")
        self.OxygenVisualValidationWidget = MatplotlibWidget(self.OxygenTab)
        self.OxygenVisualValidationWidget.setGeometry(QtCore.QRect(840, 60, 600, 480))
        self.OxygenVisualValidationWidget.setObjectName("OxygenVisualValidationWidget")
        self.OxygenLabel = QtWidgets.QLabel(self.OxygenTab)
        self.OxygenLabel.setGeometry(QtCore.QRect(110, 590, 611, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.OxygenLabel.setFont(font)
        self.OxygenLabel.setText("")
        self.OxygenLabel.setObjectName("OxygenLabel")
        self.CrossValidationLabel = QtWidgets.QLabel(self.OxygenTab)
        self.CrossValidationLabel.setGeometry(QtCore.QRect(830, 590, 611, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.CrossValidationLabel.setFont(font)
        self.CrossValidationLabel.setText("")
        self.CrossValidationLabel.setObjectName("CrossValidationLabel")
        self.tabWidget.addTab(self.OxygenTab, "")
        self.EfficiencyTab = QtWidgets.QWidget()
        self.EfficiencyTab.setObjectName("EfficiencyTab")
        self.EfficiencyVisualWidget = MatplotlibWidget(self.EfficiencyTab)
        self.EfficiencyVisualWidget.setGeometry(QtCore.QRect(120, 60, 600, 480))
        self.EfficiencyVisualWidget.setObjectName("EfficiencyVisualWidget")
        self.EfficiencyVisualValidationWidget = MatplotlibWidget(self.EfficiencyTab)
        self.EfficiencyVisualValidationWidget.setGeometry(QtCore.QRect(840, 60, 600, 480))
        self.EfficiencyVisualValidationWidget.setObjectName("EfficiencyVisualValidationWidget")
        self.EfficiencyVisualBtn = QtWidgets.QPushButton(self.EfficiencyTab)
        self.EfficiencyVisualBtn.setGeometry(QtCore.QRect(720, 680, 111, 31))
        self.EfficiencyVisualBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.EfficiencyVisualBtn.setObjectName("EfficiencyVisualBtn")
        self.EfficiencyLabel = QtWidgets.QLabel(self.EfficiencyTab)
        self.EfficiencyLabel.setGeometry(QtCore.QRect(110, 590, 611, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.EfficiencyLabel.setFont(font)
        self.EfficiencyLabel.setText("")
        self.EfficiencyLabel.setObjectName("EfficiencyLabel")
        self.EfficiencyCrossValidationLabel = QtWidgets.QLabel(self.EfficiencyTab)
        self.EfficiencyCrossValidationLabel.setGeometry(QtCore.QRect(830, 590, 611, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.EfficiencyCrossValidationLabel.setFont(font)
        self.EfficiencyCrossValidationLabel.setText("")
        self.EfficiencyCrossValidationLabel.setObjectName("EfficiencyCrossValidationLabel")
        self.tabWidget.addTab(self.EfficiencyTab, "")
        self.EfficiencyImproveTab = QtWidgets.QWidget()
        self.EfficiencyImproveTab.setObjectName("EfficiencyImproveTab")
        self.EfficiencyImproveWidget = MatplotlibWidget(self.EfficiencyImproveTab)
        self.EfficiencyImproveWidget.setGeometry(QtCore.QRect(120, 70, 600, 480))
        self.EfficiencyImproveWidget.setObjectName("EfficiencyImproveWidget")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.EfficiencyImproveTab)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(760, 70, 785, 481))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.EfficiencyImproveGridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.EfficiencyImproveGridLayout.setContentsMargins(0, 0, 0, 0)
        self.EfficiencyImproveGridLayout.setObjectName("EfficiencyImproveGridLayout")
        self.xiaolv_after_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.xiaolv_after_label.setText("")
        self.xiaolv_after_label.setAlignment(QtCore.Qt.AlignCenter)
        self.xiaolv_after_label.setObjectName("xiaolv_after_label")
        self.EfficiencyImproveGridLayout.addWidget(self.xiaolv_after_label, 4, 2, 1, 1)
        self.xiaolvlabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.xiaolvlabel.setObjectName("xiaolvlabel")
        self.EfficiencyImproveGridLayout.addWidget(self.xiaolvlabel, 4, 0, 1, 1)
        self.geimeiliang_after_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.geimeiliang_after_label.setText("")
        self.geimeiliang_after_label.setAlignment(QtCore.Qt.AlignCenter)
        self.geimeiliang_after_label.setObjectName("geimeiliang_after_label")
        self.EfficiencyImproveGridLayout.addWidget(self.geimeiliang_after_label, 3, 2, 1, 1)
        self.xiaolv_before_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.xiaolv_before_label.setText("")
        self.xiaolv_before_label.setAlignment(QtCore.Qt.AlignCenter)
        self.xiaolv_before_label.setObjectName("xiaolv_before_label")
        self.EfficiencyImproveGridLayout.addWidget(self.xiaolv_before_label, 4, 1, 1, 1)
        self.ercifenglianglabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.ercifenglianglabel.setObjectName("ercifenglianglabel")
        self.EfficiencyImproveGridLayout.addWidget(self.ercifenglianglabel, 2, 0, 1, 1)
        self.ercifengliang_before_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.ercifengliang_before_label.setText("")
        self.ercifengliang_before_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ercifengliang_before_label.setObjectName("ercifengliang_before_label")
        self.EfficiencyImproveGridLayout.addWidget(self.ercifengliang_before_label, 2, 1, 1, 1)
        self.youhuaqianlabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.youhuaqianlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.youhuaqianlabel.setObjectName("youhuaqianlabel")
        self.EfficiencyImproveGridLayout.addWidget(self.youhuaqianlabel, 1, 1, 1, 1)
        self.youhuahoulabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.youhuahoulabel.setAlignment(QtCore.Qt.AlignCenter)
        self.youhuahoulabel.setObjectName("youhuahoulabel")
        self.EfficiencyImproveGridLayout.addWidget(self.youhuahoulabel, 1, 2, 1, 1)
        self.Advicelabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.Advicelabel.setObjectName("Advicelabel")
        self.EfficiencyImproveGridLayout.addWidget(self.Advicelabel, 0, 1, 1, 1)
        self.geimeiliang_before_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.geimeiliang_before_label.setText("")
        self.geimeiliang_before_label.setAlignment(QtCore.Qt.AlignCenter)
        self.geimeiliang_before_label.setObjectName("geimeiliang_before_label")
        self.EfficiencyImproveGridLayout.addWidget(self.geimeiliang_before_label, 3, 1, 1, 1)
        self.ercifengliang_after_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.ercifengliang_after_label.setText("")
        self.ercifengliang_after_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ercifengliang_after_label.setObjectName("ercifengliang_after_label")
        self.EfficiencyImproveGridLayout.addWidget(self.ercifengliang_after_label, 2, 2, 1, 1)
        self.geimeilianglabel = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.geimeilianglabel.setObjectName("geimeilianglabel")
        self.EfficiencyImproveGridLayout.addWidget(self.geimeilianglabel, 3, 0, 1, 1)
        self.EfficiencyImproveBtn = QtWidgets.QPushButton(self.EfficiencyImproveTab)
        self.EfficiencyImproveBtn.setGeometry(QtCore.QRect(644, 669, 251, 51))
        self.EfficiencyImproveBtn.setObjectName("EfficiencyImproveBtn")
        self.EfficiencyImproveLabel = QtWidgets.QLabel(self.EfficiencyImproveTab)
        self.EfficiencyImproveLabel.setGeometry(QtCore.QRect(124, 590, 591, 51))
        self.EfficiencyImproveLabel.setText("")
        self.EfficiencyImproveLabel.setObjectName("EfficiencyImproveLabel")
        self.tabWidget.addTab(self.EfficiencyImproveTab, "")
        self.LogTab = QtWidgets.QWidget()
        self.LogTab.setObjectName("LogTab")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.LogTab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(19, 19, 1201, 781))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.gridLayoutWidget_2)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 1193, 777))
        self.scrollAreaWidgetContents_3.setMinimumSize(QtCore.QSize(0, 777))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.LogTextEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents_3)
        self.LogTextEdit.setGeometry(QtCore.QRect(-7, -1, 1211, 781))
        self.LogTextEdit.setMinimumSize(QtCore.QSize(0, 777))
        self.LogTextEdit.setObjectName("LogTextEdit")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout_2.addWidget(self.scrollArea, 0, 1, 1, 1)
        self.tabWidget.addTab(self.LogTab, "")

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(1)
        self.OxygenVisualBtn.clicked.connect(Form.OxygenVisualPlot)
        self.DataMiningBtn.clicked.connect(Form.Calculate)
        self.InputBtn.clicked.connect(Form.OpenFile)
        self.DataVisualBtn.clicked.connect(Form.DataVisualPlot)
        self.HeatMapBtn.clicked.connect(Form.HeapMapPlot)
        self.EfficiencyImproveBtn.clicked.connect(Form.EfficiencyImprove)
        self.EfficiencyVisualBtn.clicked.connect(Form.EfficiencyVisualPlot)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "火电关键参数软测量及效率优化系统"))
        Form.setWhatsThis(_translate("Form", "<html><head/><body><p><br/></p></body></html>"))
        self.InputBtn.setText(_translate("Form", "导入数据"))
        self.DataMiningBtn.setText(_translate("Form", "数据预处理"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DataMiningTab), _translate("Form", "数据预处理"))
        self.XLabel.setText(_translate("Form", "X轴："))
        self.XComboBox.setItemText(0, _translate("Form", "sample"))
        self.XComboBox.setItemText(1, _translate("Form", "给煤量"))
        self.XComboBox.setItemText(2, _translate("Form", "排烟温度"))
        self.XComboBox.setItemText(3, _translate("Form", "主蒸汽压力"))
        self.XComboBox.setItemText(4, _translate("Form", "炉膛出口烟温度"))
        self.XComboBox.setItemText(5, _translate("Form", "主蒸汽温度"))
        self.XComboBox.setItemText(6, _translate("Form", "机组负荷"))
        self.XComboBox.setItemText(7, _translate("Form", "给水温度"))
        self.XComboBox.setItemText(8, _translate("Form", "给水流量"))
        self.XComboBox.setItemText(9, _translate("Form", "一次风量"))
        self.XComboBox.setItemText(10, _translate("Form", "二次风量"))
        self.XComboBox.setItemText(11, _translate("Form", "二次风温"))
        self.XComboBox.setItemText(12, _translate("Form", "含氧量"))
        self.XComboBox.setItemText(13, _translate("Form", "效率"))
        self.YLabel.setText(_translate("Form", "Y轴："))
        self.YComboBox.setItemText(0, _translate("Form", "含氧量"))
        self.YComboBox.setItemText(1, _translate("Form", "效率"))
        self.YComboBox.setItemText(2, _translate("Form", "给煤量"))
        self.YComboBox.setItemText(3, _translate("Form", "排烟温度"))
        self.YComboBox.setItemText(4, _translate("Form", "主蒸汽压力"))
        self.YComboBox.setItemText(5, _translate("Form", "炉膛出口烟温度"))
        self.YComboBox.setItemText(6, _translate("Form", "主蒸汽温度"))
        self.YComboBox.setItemText(7, _translate("Form", "机组负荷"))
        self.YComboBox.setItemText(8, _translate("Form", "给水温度"))
        self.YComboBox.setItemText(9, _translate("Form", "给水流量"))
        self.YComboBox.setItemText(10, _translate("Form", "一次风量"))
        self.YComboBox.setItemText(11, _translate("Form", "二次风量"))
        self.YComboBox.setItemText(12, _translate("Form", "二次风温"))
        self.DataVisualBtn.setText(_translate("Form", "画图"))
        self.DataVisualLabel.setText(_translate("Form", "特殊类型图："))
        self.HeatMapBtn.setText(_translate("Form", "1. HeatMap"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DataVisualTab), _translate("Form", "数据可视化"))
        self.OxygenVisualBtn.setText(_translate("Form", "预测"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.OxygenTab), _translate("Form", "烟气含氧量软测量"))
        self.EfficiencyVisualBtn.setText(_translate("Form", "预测"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.EfficiencyTab), _translate("Form", "效率软测量"))
        self.xiaolvlabel.setText(_translate("Form", "效率："))
        self.ercifenglianglabel.setText(_translate("Form", "二次风量："))
        self.youhuaqianlabel.setText(_translate("Form", "优化前"))
        self.youhuahoulabel.setText(_translate("Form", "优化后"))
        self.Advicelabel.setText(_translate("Form", "控制量优化建议"))
        self.geimeilianglabel.setText(_translate("Form", "给煤量："))
        self.EfficiencyImproveBtn.setText(_translate("Form", "开始优化"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.EfficiencyImproveTab), _translate("Form", "效率优化"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.LogTab), _translate("Form", "系统日志"))

from MatplotlibWidget import MatplotlibWidget
