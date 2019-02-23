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
        Form.resize(1610, 936)
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
        self.DataVisualWidget.setGeometry(QtCore.QRect(19, 79, 1531, 731))
        self.DataVisualWidget.setObjectName("DataVisualWidget")
        self.DataVisualBtn = QtWidgets.QPushButton(self.DataVisualTab)
        self.DataVisualBtn.setGeometry(QtCore.QRect(50, 30, 111, 31))
        self.DataVisualBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.DataVisualBtn.setObjectName("DataVisualBtn")
        self.tabWidget.addTab(self.DataVisualTab, "")
        self.OxygenTab = QtWidgets.QWidget()
        self.OxygenTab.setObjectName("OxygenTab")
        self.OxygenVisualWidget = MatplotlibWidget(self.OxygenTab)
        self.OxygenVisualWidget.setGeometry(QtCore.QRect(20, 70, 1531, 731))
        self.OxygenVisualWidget.setObjectName("OxygenVisualWidget")
        self.OxygenVisualBtn = QtWidgets.QPushButton(self.OxygenTab)
        self.OxygenVisualBtn.setGeometry(QtCore.QRect(50, 30, 111, 31))
        self.OxygenVisualBtn.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.OxygenVisualBtn.setObjectName("OxygenVisualBtn")
        self.tabWidget.addTab(self.OxygenTab, "")
        self.EfficiencyTab = QtWidgets.QWidget()
        self.EfficiencyTab.setObjectName("EfficiencyTab")
        self.tabWidget.addTab(self.EfficiencyTab, "")
        self.EfficiencyImproveTab = QtWidgets.QWidget()
        self.EfficiencyImproveTab.setObjectName("EfficiencyImproveTab")
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
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, -4, 1193, 777))
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
        self.tabWidget.setCurrentIndex(2)
        self.InputBtn.clicked.connect(Form.OpenFile)
        self.DataMiningBtn.clicked.connect(Form.Calculate)
        self.DataVisualBtn.clicked.connect(Form.DataVisualPlot)
        self.OxygenVisualBtn.clicked.connect(Form.OxygenVisualPlot)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "火电关键参数软测量及效率优化系统"))
        Form.setWhatsThis(_translate("Form", "<html><head/><body><p><br/></p></body></html>"))
        self.InputBtn.setText(_translate("Form", "导入数据"))
        self.DataMiningBtn.setText(_translate("Form", "数据预处理"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DataMiningTab), _translate("Form", "数据预处理"))
        self.DataVisualBtn.setText(_translate("Form", "画图"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.DataVisualTab), _translate("Form", "数据可视化"))
        self.OxygenVisualBtn.setText(_translate("Form", "画图"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.OxygenTab), _translate("Form", "烟气含氧量软测量"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.EfficiencyTab), _translate("Form", "效率软测量"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.EfficiencyImproveTab), _translate("Form", "效率优化"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.LogTab), _translate("Form", "系统日志"))

from MatplotlibWidget import MatplotlibWidget
