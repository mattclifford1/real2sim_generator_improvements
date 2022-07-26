import time
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QGridLayout, QLabel, QSlider, QComboBox, QVBoxLayout


class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        layout = QVBoxLayout(self)
        self.label = QLabel(self)
        layout.addWidget(self.label)
        self.buttonStart = QPushButton('Start', self)
        self.buttonStart.clicked.connect(self.handleStart)
        layout.addWidget(self.buttonStart)
        self.buttonStop = QPushButton('Stop', self)
        self.buttonStop.clicked.connect(self.handleStop)
        layout.addWidget(self.buttonStop)
        self._running = False

    def handleStart(self):
        self.buttonStart.setDisabled(True)
        self._running = True
        while self._running:
            self.label.setText(str(time.time()))
            # qApp.processEvents()
            QApplication.processEvents()
            time.sleep(0.05)
        self.buttonStart.setDisabled(False)

    def handleStop(self):
        self._running = False

if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 200, 100)
    window.show()
    sys.exit(app.exec())
