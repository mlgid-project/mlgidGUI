from PyQt5.QtWidgets import QWidget, QLineEdit, QFormLayout
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import pyqtSignal, QRegExp

class CIFFileWidget(QWidget):
    sigSetCIFFile = pyqtSignal(float)

    def __init__(self):
        QWidget.__init__(self)
        self.line_edit = QLineEdit()
        rx = QRegExp("^[A-Za-z0-9]*$")
        validator = QRegExpValidator(rx, self.line_edit)
        self.line_edit.setValidator(validator)
        flo = QFormLayout()
        flo.addRow("CIF file name", self.line_edit)
        self.setLayout(flo)