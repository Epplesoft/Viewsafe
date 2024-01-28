from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QSlider, QLabel, QCheckBox, QVBoxLayout, QHBoxLayout, QFileDialog, QListWidget, QPushButton, QSplitter, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QSize
import cv2
import numpy as np
import os
import sys


# Custom Frame class to make sure each widget has square dimensions
class SquareFrame(QtWidgets.QFrame):
    def __init__(self, label=None, parent=None):
        super(SquareFrame, self).__init__(parent)
        self.label = label

    # Override the sizeHint method to adjust dimensions to square
    def sizeHint(self):
        if self.label:
            metrics = self.label.fontMetrics()
            text_width = metrics.horizontalAdvance(self.label.text())
            padding = 10
            dimension = text_width + padding
        else:
            dimension = 100  # Default value if no label provided
        return QtCore.QSize(dimension, dimension)


class ClickableLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.parent = parent
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_menu_show)

    def right_menu_show(self, pos):
        context_menu = QtWidgets.QMenu(self)
        save_action = context_menu.addAction('Save Image As')
        save_action.triggered.connect(self.parent.save_image)
        context_menu.exec_(QtGui.QCursor.pos())


# Main GUI Window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, image_paths=None):
        super(MainWindow, self).__init__()


        #For right-click 'Save As'
        self.image_label = ClickableLabel(self)

        #Set the window icon
        dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
        icon_path = os.path.join(dir_path, 'icon.ico')  # Join directory with icon filename
        self.setWindowIcon(QtGui.QIcon(icon_path))

        #Set the window title
        self.setWindowTitle("Image Viewer")
        self.setMinimumSize(600, 500)  # Adjust the numbers as needed


        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QtWidgets.QHBoxLayout(self.main_widget)
        
        # QSplitter for resizable layout
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # List of images and button to select folder
        self.image_list = QListWidget()
        self.image_list.setFixedWidth(200)
        splitter.addWidget(self.image_list)  

        self.folder_button = QPushButton("Select Folder")  
        self.folder_button.clicked.connect(self.select_folder)  

        self.image_list.currentItemChanged.connect(self.image_selected)  

        image_display_widget = QtWidgets.QWidget()
        self.image_layout = QVBoxLayout(image_display_widget)

        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label) 
        self.image_layout.addWidget(self.folder_button)
        splitter.addWidget(image_display_widget)

        self.layout.addWidget(splitter)

        self.pixelation_layout = QVBoxLayout()  
        self.layout.addLayout(self.pixelation_layout)  

        self.pixelation_layout.addWidget(QLabel("<b>Pixelation</b>"))

        self.max_pixelation_label = QLabel("Max")  
        self.pixelation_layout.addWidget(self.max_pixelation_label)  

        self.pixelation_slider = QSlider(Qt.Vertical, self)
        self.pixelation_layout.addWidget(self.pixelation_slider)
        self.pixelation_slider.setRange(1, 20)  
        self.pixelation_slider.setValue(20)  
        self.pixelation_slider.valueChanged.connect(self.update_image)

        self.min_pixelation_label = QLabel("None")  
        self.pixelation_layout.addWidget(self.min_pixelation_label)  

        self.original_image = None

        # Grayscale
        self.grayscale_layout = QVBoxLayout()
        self.layout.addLayout(self.grayscale_layout)

        self.grayscale_label = QLabel("<b>Grayscale</b>")
        self.grayscale_check = QCheckBox()
        self.grayscale_check.setChecked(True)
        self.grayscale_check.stateChanged.connect(self.update_image)

        self.grayscale_label.setStyleSheet("border: none")
        self.grayscale_check.setStyleSheet("border: none")

        self.grayscale_widget = SquareFrame(label=self.grayscale_label)

        self.grayscale_inner_layout = QVBoxLayout(self.grayscale_widget)

        self.grayscale_inner_layout.addWidget(self.grayscale_label)
        self.grayscale_inner_layout.addWidget(self.grayscale_check, alignment=Qt.AlignCenter)

        self.grayscale_widget.setStyleSheet("border: 1px solid lightgray")

        self.grayscale_layout.addStretch(1)  
        self.grayscale_layout.addWidget(self.grayscale_widget)
        self.grayscale_layout.addStretch(1)

        self.image_directory = ""  # Holds the path of the currently selected directory
        self.image_files = []  # Holds the names of all image files in the directory

        if image_paths is not None:  # If there are command-line arguments
            if os.path.isdir(image_paths[0]):  # If the first argument is a directory
                self.load_folder(image_paths[0])  # Load all images from the directory
            else:
                self.add_images(image_paths)  # Else load the individual image files
        else:
            # Default image displayed when no arguments provided
            if getattr(sys, 'frozen', False):  # PyInstaller creates a temp folder and stores path in _MEIPASS
                bundle_dir = sys._MEIPASS
            else:
                bundle_dir = os.path.dirname(os.path.abspath(__file__))

            default_image = os.path.join(bundle_dir, "default_image.png")  # Now the path is always correct
            if os.path.isfile(default_image):
                self.add_images([default_image])


    def resizeEvent(self, event):
        self.update_image()
        super(MainWindow, self).resizeEvent(event)


    def load_folder(self, folder):
        self.image_directory = folder
        # Filter for common image extensions
        self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        self.image_list.clear()
        # Populate the QListWidget
        for i, file in enumerate(self.image_files, start=1):  
            self.image_list.addItem(str(i) + " - " + file) # Concatenate index and filename
        self.showMaximized()  # Full-screen after folder selection
        # Select the first item in the list
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            QTimer.singleShot(100, lambda: self.load_image(os.path.join(self.image_directory, self.image_files[0])))  # Delay the loading of the image

    # Method to select a folder and populate the list with images
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")  # Open a QFileDialog
        if folder:
            self.load_folder(folder)  # Load the folder if one is selected

    def add_images(self, image_paths):
        self.image_files = image_paths
        self.image_list.clear()
        for i, file in enumerate(self.image_files, start=1):
            self.image_list.addItem(str(i) + " - " + os.path.basename(file))
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            QTimer.singleShot(100, lambda: self.load_image(self.image_files[0]))

    # Method to load an image when selected from the list
    def image_selected(self, item):
        if item is not None: # Check if item is not None (this could happen if list is cleared)
            index = int(item.text().split(" - ")[0]) - 1
            image_path = os.path.join(self.image_directory, self.image_files[index]) if self.image_directory else self.image_files[index]
            self.load_image(image_path)

    # Method to load an image and display it
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.update_image()

    def update_image(self):
        if self.original_image is not None:
            image = np.copy(self.original_image)
            if self.grayscale_check.isChecked():
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if self.pixelation_slider.value() > 1:
                h, w, _ = image.shape
                k = max(1, min(h, w, int(min(h, w) / ((24 - self.pixelation_slider.value() + 1)**2.3 / 6))))
                image = cv2.resize(cv2.resize(image, (w // k, h // k)), (w, h), interpolation=cv2.INTER_NEAREST)

            self.processed_image = image  # Store the processed image for saving

            height, width = image.shape[:2]
            bytes_per_line = 3 * width
            format = QtGui.QImage.Format_RGB888

            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, format).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            cv2.imwrite(file_path, self.processed_image)