import sys
from PyQt5 import QtWidgets, QtGui, QtCore
 
 
class ClassTextAreaWrite(QtWidgets.QWidget):
 
    def __init__(self):
        super(ClassTextAreaWrite, self).__init__()
        self.init_ui()
        self.bind_trigger()
 
    def init_ui(self):
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle("PushButtonClicked")
        button = QtWidgets.QPushButton(text="Start", parent=self)
        self.button = button
        timer = QtCore.QTimer()
        self.timer = timer
 
        view_area = QtWidgets.QPlainTextEdit(parent=self)
        view_area.setGeometry(10, 60, 280, 100)
        self.view_area = view_area
 
    def write(self, text):
        """"""
        cursor = self.view_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text + "\n")
        self.view_area.setTextCursor(cursor)
        self.view_area.ensureCursorVisible()
 
    def bind_trigger(self):
        # Button clicked trigger
        self.button.clicked.connect(lambda: self.timer.start(1))
        self.timer.timeout.connect(self.clicked_button)
 
    def clicked_button(self):
        self.write("Write Num: {0}".format(self.timer.interval()))
        self.timer.setInterval(self.timer.interval() + 1)
 
 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    push_button_clicked = ClassTextAreaWrite()
    push_button_clicked.show()
    sys.exit(app.exec_())