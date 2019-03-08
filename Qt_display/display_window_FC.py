from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSlot
# from Qt_display.display_window_UI import Ui_MainWindow
from Qt_display.display_window_UI import Ui_MainWindow


class Link_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # use the super init
        super(Link_MainWindow, self).__init__()
        # self.setupUi(self)

    @pyqtSlot()
    def on_clear_button_clicked(self):
        self.display_txtbox.setText('')
        QMessageBox.information(self, '提示', '清除成功')

    @pyqtSlot()
    def on_input_button_clicked(self):
        append_str = self.input_txtbox.text()
        self.display_txtbox.append(append_str)

    @pyqtSlot(int)
    def on_test_dial_valueChanged(self, value):
        self.test_slider.setValue(value)

    @pyqtSlot(int)
    def on_test_slider_valueChanged(self, value):
        self.test_dial.setValue(value)

    @pyqtSlot()
    def on_quit_button_clicked(self):
        decision = QMessageBox.question(self, 'Warning', 'Quit this program?',
                                        QMessageBox.Yes | QMessageBox.No)
        if decision == QMessageBox.Yes:
            self.close()


class Func_MainWindow(Link_MainWindow):
    def __init__(self):
        super(Func_MainWindow, self).__init__()

    # send the message to the txtbox
    def send_to_display(self, message):
        self.display_txtbox.append(message)


if __name__ == '__main__':
    # before run this test for checking the UI, you should uncomment the line 12 first.
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Main_window = Func_MainWindow()
    Main_window.show()
    sys.exit(app.exec_())
