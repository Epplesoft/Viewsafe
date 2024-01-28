from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QSlider, QLabel, QCheckBox, QVBoxLayout, QHBoxLayout, QFileDialog, QListWidget, QPushButton, QSplitter, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QSize
import cv2
import numpy as np
import os
import sys

from main_window import MainWindow

# At the beginning of your script
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle/exe, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow(sys.argv[1:] if len(sys.argv) > 1 else None)  # Pass command-line arguments if there are any
    main_window.show()
    sys.exit(app.exec_())
