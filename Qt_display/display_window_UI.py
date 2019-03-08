# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'display_window_UI.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        MainWindow.setFont(font)
        MainWindow.setDockNestingEnabled(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input_txtbox = QtWidgets.QLineEdit(self.centralwidget)
        self.input_txtbox.setGeometry(QtCore.QRect(10, 610, 320, 26))
        self.input_txtbox.setObjectName("input_txtbox")
        self.clear_button = QtWidgets.QPushButton(self.centralwidget)
        self.clear_button.setGeometry(QtCore.QRect(10, 660, 100, 30))
        self.clear_button.setObjectName("clear_button")
        self.input_button = QtWidgets.QPushButton(self.centralwidget)
        self.input_button.setGeometry(QtCore.QRect(120, 660, 100, 30))
        self.input_button.setObjectName("input_button")
        self.quit_button = QtWidgets.QPushButton(self.centralwidget)
        self.quit_button.setGeometry(QtCore.QRect(230, 660, 100, 30))
        self.quit_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.quit_button.setObjectName("quit_button")
        self.display_txtbox = QtWidgets.QTextBrowser(self.centralwidget)
        self.display_txtbox.setGeometry(QtCore.QRect(10, 10, 320, 591))
        self.display_txtbox.setObjectName("display_txtbox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_txtbox.setText(_translate("MainWindow", "在这里输入"))
        self.clear_button.setText(_translate("MainWindow", "Clear"))
        self.input_button.setText(_translate("MainWindow", "Input"))
        self.quit_button.setText(_translate("MainWindow", "Quit"))
        self.display_txtbox.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))

