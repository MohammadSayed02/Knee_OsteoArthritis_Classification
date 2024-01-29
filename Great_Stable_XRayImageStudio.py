'''
 * ________________________________________ Radiographic Image Studio ________________________________________________
 *
 *  Features Extraction and Visuallization - Conventional CAD - Ai Automated CAD
 *  Created on: Friday Sep 15 2023
 *  Author    : Mohammad Sayed Zaky - BME Team 13
 '''
 
#  _____________________________________________ Libraries ____________________________________________________________

from PyQt5.QtWidgets import QApplication, QMainWindow, QToolTip, QFileDialog, QLabel,QMessageBox,QVBoxLayout,QPushButton, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon, QFont, QPainter, QPainterPath, QRegion, QPen, QPolygonF, QBrush, QPainter
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QSize, QPoint, QPointF
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import subprocess
import sys
import cv2
import os
import glob
import time
import csv
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from ultralytics import YOLO
import torch
from Models.modelCNN import SimpleCNN
from joblib import load
#  _____________________________________________ Splash Screen ____________________________________________________________
class SplashScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Splash Screen")
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        # self.resize(1000, 800)
        Feature_Extraction_and_Visualization_Screen.image = None
        Feature_Extraction_and_Visualization_Screen.image_label = None
        self.background_label = QLabel(self)
        self.pixmap = QPixmap("imgs/blackSplashScreen.png")
        self.background_label.setPixmap(self.pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, 1920, 1080)
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(QPixmap("imgs/LogoSplashScreen_Original.png"))
        self.logo_label.setScaledContents(True)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("color: black; font-size: 24px; background: rgba(0, 0, 0, 100);")
        self.logo_label.setGeometry(625, 125, 700, 700)
        self.loading_label = QLabel(self)
        self.loading_label.setPixmap(QPixmap("imgs/loading_label.png"))
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setGeometry(825, 850, 300, 10)
        
    #  _____________________________________________ Including Models ________________________________________________
        
        self.model =YOLO("Models/YOLO_ROI detection.pt")
        self.model_instance_Segmentation =YOLO("Models/YOLO_Instance Segmentation.pt")
        self.model_YOLO_CenteralReg =YOLO("Models/YOLO_CenteralReg.pt")
        self.model_YOLO_TibialWidth =YOLO("Models/YOLO_TibialWidth.pt")
        self.knee_Literality_model = SimpleCNN()
        self.knee_Literality_model.load_state_dict(torch.load('Models/Knee_Literality.pth'))
        self.knee_Literality_model.eval()
                                 # ___________ Conventional CAD Model ____________#
                                    
        self.Random_Forest_Normale_Binary = load('Models/models/Random_Forest_Normale_Binary.pkl')
        self.scaler_Normalize_Binary = load('Models/models/scaler_Normalize_Binary.pkl')
        
        
        self.Random_Forest_Normalize_3_Class = load('Models/models/Random_Forest_Normalize_3_Class.pkl')
        self.scaler_Normalize_3_Class = load('Models/models/scaler_Normalize_3_Class.pkl')
        
        
        self.Random_Forest_Normalize_4_Classes = load('Models/models/Random_Forest_Normalize_4_Classes.pkl')
        self.scaler_Normalize_4_Class = load('Models/models/scaler_Normalize_4_Class.pkl')
                
        self.SVM_Normalize_1_2_Classes = load('Models/models/SVM_Normalize_1_2_Classes.pkl')
        self.scaler_Normalize_1_2_Class = load('Models/models/scaler_Normalize_1_2_Class.pkl')
        
        
        self.Random_Forest_Normalize_3_4Classes = load('Models/models/Random_Forest_Normalize_3_4Classes.pkl')
        self.scaler_Normalize_3_4_Class = load('Models/models/scaler_Normalize_3_4_Class.pkl')
    
        self.Random_Forest_Normalize_5_Class = load('Models/models/Random_Forest_Normalize_5_Class.pkl')
        self.scaler_Normalize_5_Class = load('Models/models/scaler_Normalize_5_Class.pkl')
        
        
        
        
        
        
        # new_features_Raw = [17, 12, 13, 19, 18, 14, 32, 33, 29, 30, 25, 23, 24, 57, 46, 53, 44, 45, 41, 37]
        # new_features_Raw = data_Raw_Normalize_Class[['Medial Area (Squared Pixel)', 'Tibial_Medial_ratio',
        #                                     'Tibial_Central_ratio', 'Lateral Area (Squared Pixel)', 
        #                                     'Central Area (Squared Pixel)', 'Tibial_Lateral_ratio'
        #                                     ,'ASM', 'max_probability', 'correlation', 'homogeneity' , 
        #                                     'Skewness', 'Mean', 'Sigma' ,
        #                                     'LTP_Entropy', 'LTP_0', 'LTP_7' ,
        #                                     'LBP_Variance', 'LBP_Entropy', 'LBP_7', 'LBP_3'
                                            
        #                                     ]]
        
                                # ___________ Ai Automated CAD Model ____________#
    
    #  _______________________________________________________________________________________________________________

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close_splash)
        self.timer.start(1000)
        

    def close_splash(self):
        self.close()
        # ___________________________________resize to Full Screen____________________________________________________
    def resizeEvent(self, event):
        pixmap = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(pixmap)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
#  ____________________________________________ CustomMessageBox _____________________________________________________
class CustomMessageBox(QMessageBox):
    def __init__(self):
        super(CustomMessageBox, self).__init__()
        self.setWindowTitle('Exit!')
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setText("You are about to Exit!   Do you want to Leave?")
        quit_button = self.addButton("Exit", QMessageBox.AcceptRole)
        self.addButton("Stay", QMessageBox.RejectRole)
        quit_button.setStyleSheet("color: black;")
        self.setStyleSheet("QMessageBox { background-color: white; color: white; }")

#  _____________________________________________ ImageLabel __________________________________________________________

class ImageLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.crop_rect = QRect()
        self.mouse_pressed = False
        self.setStyleSheet("color: white;border-radius:100px;background-color:#000000")

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.mouse_pressed:
            painter = QPainter(self)
            color = QColor(0, 255, 0)
            color.setAlpha(255)
            pen = QPen(color)
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(self.crop_rect)
            
#  _____________________________________________ RoundImageLabel __________________________________________________________

class RoundImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.setAlignment(Qt.AlignCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.pixmap():
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), 100, 100)
            region = QRegion(path.toFillPolygon().toPolygon())
            self.setMask(region)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, self.pixmap())
            
#  _____________________________________________ AI_Automated_CAD Screen ____________________________________________________
            
class AI_Automated_CAD_Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setWindowTitle("Computer aided Diagnosis")
        # self.setFixedSize(1000, 800)
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        self.splash_screen = SplashScreen()
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen()
        self.Conventional_CAD_Screen = Conventional_CAD_Screen(parent = self.parent)
# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
# ______________________________________________ show_main_content __________________________________________________
        
        self.show_main_content()

    def show_main_content(self):
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)

        self.image_label = RoundImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-image: url(imgs/Ai_Automated_CAD_Poster.png);background-repeat: no-repeat;")
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(600)
        self.image_label.setFixedHeight(600)
        self.image_label.setToolTip("Double click or Drag and drop to Add a New image")
        self.image_label2 = ImageLabel("AI Automated CAD Output")
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        font = QFont()
        font.setPointSize(20)
        self.image_label2.setFont(font)
        self.home_button = QPushButton(self)
        homeicon = QIcon("imgs/clear.png")
        self.home_button.setIcon(homeicon)
        icon_size = QSize(50, 50)
        self.home_button.setIconSize(icon_size)
        self.home_button.setStyleSheet("background-color: transparent; border: none;")
        self.home_button.clicked.connect(self.on_button_click)
        self.home_button.setToolTip("Clear")
        self.Expand_button = QPushButton(self)
        self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
        self.Expand_button.setMinimumSize(50,50)
        self.Expand_button.setMaximumSize(50,50)
        self.Expand_button.setIconSize(QSize(50,50))
        self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Expand_button.clicked.connect(self.Expand_Function)
        self.Expand_button.setToolTip("Exit Full Screen (ESC)")
        self.Restart_button = QPushButton(self)
        self.Restart_button.setIcon(QIcon('imgs/restart.png'))
        self.Restart_button.setMinimumSize(35,35)
        self.Restart_button.setMaximumSize(35,35)
        self.Restart_button.setIconSize(QSize(35,35))
        self.Restart_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Restart_button.clicked.connect(Feature_Extraction_and_Visualization_Screen.restart_code)
        self.Restart_button.setToolTip("Restart")
        self.EXIT_button = QPushButton(self)
        self.EXIT_button.setIcon(QIcon('imgs/exit.png'))
        self.EXIT_button.setMinimumSize(60,60)
        self.EXIT_button.setMaximumSize(60,60)
        self.EXIT_button.setIconSize(QSize(60,60))
        self.EXIT_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.EXIT_button.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.EXIT_Function)
        self.EXIT_button.setToolTip("EXIT")
        self.Feature_Extraction_and_Visualization_button = QPushButton(self)
        self.Feature_Extraction_and_Visualization_button.setIcon(QIcon('imgs/Feature_Extraction.png'))
        self.Feature_Extraction_and_Visualization_button.setMinimumSize(95,95)
        self.Feature_Extraction_and_Visualization_button.setMaximumSize(95,95)
        self.Feature_Extraction_and_Visualization_button.setIconSize(QSize(95,95))
        self.Feature_Extraction_and_Visualization_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Feature_Extraction_and_Visualization_button.clicked.connect(self.Feature_Extraction_and_Visualization_Mode)
        self.Feature_Extraction_and_Visualization_button.setToolTip("Return to Feature Extraction and Visualization")             
        self.Conventional_CAD_button = QPushButton(self)
        self.Conventional_CAD_button.setIcon(QIcon('imgs/Conventional_CAD_Poster.png'))
        self.Conventional_CAD_button.setMinimumSize(80,80)
        self.Conventional_CAD_button.setMaximumSize(80,80)
        self.Conventional_CAD_button.setIconSize(QSize(80,80))
        self.Conventional_CAD_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Conventional_CAD_button.clicked.connect(self.switch_to_conventional_cad)
        self.Conventional_CAD_button.setToolTip("Redirect to Conventional CAD")
        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer2 = QSpacerItem(100, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        HBoxLayout2 = QHBoxLayout()
        HBoxLayout2.addItem(spacer)
        HBoxLayout2.addWidget(self.Expand_button)
        HBoxLayout2.addWidget(self.home_button)
        HBoxLayout2.addWidget(self.Restart_button)
        HBoxLayout2.addWidget(self.EXIT_button)
        HBoxLayout2.addWidget(self.Conventional_CAD_button)
        HBoxLayout2.addWidget(self.Feature_Extraction_and_Visualization_button)
        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addItem(spacer2)
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.image_label2)
        VBoxLayout1.addLayout(HBoxLayout2)
        self.setLayout(VBoxLayout1)

# __________________________________________________ Functions _______________________________________________________

    def mouseDoubleClickEvent(self, event):
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                self.file_path = file_paths[0]
                self.image = cv2.imread(self.file_path)

                if self.image is not None:
                    height, width = self.image.shape[:2]
                    format = QImage.Format_Grayscale8 if len(self.image.shape) == 2 else QImage.Format_RGB888
                    q_img = QImage(self.image.data, width, height, self.image.strides[0], format)

                    pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    rounded_pixmap = self.create_round_image(pixmap)

                    if rounded_pixmap:
                        self.display_image(rounded_pixmap, self.image_label)
                else:
                    print("Error: Unable to load the image.")

    def create_round_image(self, pixmap):
        if pixmap.isNull():
            return None

        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.black)
        painter.setPen(Qt.transparent)
        path = QPainterPath()
        path.addRoundedRect(0, 0, size.width(), size.height(), 100, 100)
        painter.drawPath(path)
        painter.end()
        result = QPixmap(size)
        result.fill(Qt.transparent)
        painter.begin(result)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return result

    def display_image(self, img, target_label=None):
        if img is not None:
            if isinstance(img, QPixmap):
                if target_label is not None:
                    target_label.setPixmap(img)
                    target_label.setScaledContents(True)
                else:
                    self.image_label.setPixmap(img)
                    self.image_label.setScaledContents(True)
            else:
                print("\033[91mUnsupported image format.\033[0m")
                if target_label is not None:
                    target_label.setText("\033[91mUnsupported image format.\033[0m")
                else:
                    self.image_label.setText("\033[91mUnsupported image format.\033[0m")
                      
                
    def Expand_Function(self):
        if self.isFullScreen:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
            self.isFullScreen = False
            self.Expand_button.setIcon(QIcon('imgs/expand.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 70px; }")
            self.Expand_button.setToolTip("Expand")
        
        
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            self.isFullScreen = True
            self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Expand_button.setToolTip("Exit Full Screen (ESC)")
            

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen:
            self.Expand_Function()
        else:
            super().keyPressEvent(event)
            

    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        self.image_label.setStyleSheet("background-image: url(imgs/Ai_Automated_CAD_ICON.png);background-image: no-repeat;")
        self.image_label2.setText("AI Automated CAD Output")

    def switch_to_conventional_cad(self):
        self.on_button_click()
        self.parent.setCentralWidget(self.Conventional_CAD_Screen)
        self.Conventional_CAD_Screen.show()
        self.hide()

    def Feature_Extraction_and_Visualization_Mode(self):
        self.on_button_click()
        self.parent.setCentralWidget(self.Feature_Extraction_and_Visualization_Screen)
        self.Feature_Extraction_and_Visualization_Screen.show()
        self.hide()

    def closeEvent(self, event):
        msgBox = CustomMessageBox()
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            exit(0)
        else:
            event.ignore()

#  _____________________________________________ Conventional_CAD Screen ____________________________________________________
            
class Conventional_CAD_Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setWindowTitle("Computer aided Diagnosis")
        # self.setFixedSize(1000, 800)
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        
        self.splash_screen = SplashScreen()
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen()
        # self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(parent = self.parent)

# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.Features_input = []
# ______________________________________________ show_main_content __________________________________________________
        
        self.show_main_content()

    def show_main_content(self):
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        
        self.image_label = RoundImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-image: url(imgs/Conventional_CAD_Poster.png);background-repeat: no-repeat;")
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(600)
        self.image_label.setFixedHeight(600)
        self.image_label.setToolTip("Double click or Drag and drop to Add a New image")
        self.image_label2 = ImageLabel("Conventional CAD Output")
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        font = QFont()
        font.setPointSize(20)
        self.image_label2.setFont(font)
        self.home_button = QPushButton(self)
        homeicon = QIcon("imgs/clear.png")
        self.home_button.setIcon(homeicon)
        icon_size = QSize(50, 50)
        self.home_button.setIconSize(icon_size)
        self.home_button.setStyleSheet("background-color: transparent; border: none;")
        self.home_button.clicked.connect(self.on_button_click)
        self.home_button.setToolTip("Clear")
        self.Expand_button = QPushButton(self)
        self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
        self.Expand_button.setMinimumSize(50,50)
        self.Expand_button.setMaximumSize(50,50)
        self.Expand_button.setIconSize(QSize(50,50))
        self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Expand_button.clicked.connect(self.Expand_Function)
        self.Expand_button.setToolTip("Exit Full Screen (ESC)")
        self.Restart_button = QPushButton(self)
        self.Restart_button.setIcon(QIcon('imgs/restart.png'))
        self.Restart_button.setMinimumSize(35,35)
        self.Restart_button.setMaximumSize(35,35)
        self.Restart_button.setIconSize(QSize(35,35))
        self.Restart_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Restart_button.clicked.connect(Feature_Extraction_and_Visualization_Screen.restart_code)
        self.Restart_button.setToolTip("Restart")
        self.EXIT_button = QPushButton(self)
        self.EXIT_button.setIcon(QIcon('imgs/exit.png'))
        self.EXIT_button.setMinimumSize(60,60)
        self.EXIT_button.setMaximumSize(60,60)
        self.EXIT_button.setIconSize(QSize(60,60))
        self.EXIT_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.EXIT_button.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.EXIT_Function)
        self.EXIT_button.setToolTip("EXIT")
        self.Feature_Extraction_and_Visualization_button = QPushButton(self)
        self.Feature_Extraction_and_Visualization_button.setIcon(QIcon('imgs/Feature_Extraction.png'))
        self.Feature_Extraction_and_Visualization_button.setMinimumSize(95,95)
        self.Feature_Extraction_and_Visualization_button.setMaximumSize(95,95)
        self.Feature_Extraction_and_Visualization_button.setIconSize(QSize(95,95))
        self.Feature_Extraction_and_Visualization_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Feature_Extraction_and_Visualization_button.clicked.connect(self.Feature_Extraction_and_Visualization_Mode)
        self.Feature_Extraction_and_Visualization_button.setToolTip("Return to Feature Extraction and Visualization")
        self.Ai_Automated_CAD_button = QPushButton(self)
        self.Ai_Automated_CAD_button.setIcon(QIcon('imgs/Ai_Automated_CAD_ICON.png'))
        self.Ai_Automated_CAD_button.setMinimumSize(95,95)
        self.Ai_Automated_CAD_button.setMaximumSize(95,95)
        self.Ai_Automated_CAD_button.setIconSize(QSize(95,95))
        self.Ai_Automated_CAD_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Ai_Automated_CAD_button.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.AI_Automated_CAD2)
        self.Ai_Automated_CAD_button.setToolTip("Redirect to Ai Automated CAD")
        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer2 = QSpacerItem(100, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        HBoxLayout2 = QHBoxLayout()
        HBoxLayout2.addItem(spacer)
        HBoxLayout2.addWidget(self.Expand_button)
        HBoxLayout2.addWidget(self.home_button)
        HBoxLayout2.addWidget(self.Restart_button)
        HBoxLayout2.addWidget(self.EXIT_button)
        HBoxLayout2.addWidget(self.Ai_Automated_CAD_button)
        HBoxLayout2.addWidget(self.Feature_Extraction_and_Visualization_button)
        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addItem(spacer2)
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.image_label2)
        VBoxLayout1.addLayout(HBoxLayout2)
        self.setLayout(VBoxLayout1)

# __________________________________________________ Functions _______________________________________________________

    def mouseDoubleClickEvent(self, event):
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")

            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                self.file_path = file_paths[0]
                self.image = cv2.imread(self.file_path)
                    
                if self.image is not None:
                    height, width = self.image.shape[:2]
                    format = QImage.Format_Grayscale8 if len(self.image.shape) == 2 else QImage.Format_RGB888
                    q_img = QImage(self.image.data, width, height, self.image.strides[0], format)

                    pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    rounded_pixmap = self.create_round_image(pixmap)

                    if rounded_pixmap:
                        self.display_image(rounded_pixmap, self.image_label)
                        self.carry_out()
                else:
                    print("Error: Unable to load the image.")

    def create_round_image(self, pixmap):
        if pixmap.isNull():
            return None
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.black)
        painter.setPen(Qt.transparent)
        path = QPainterPath()
        path.addRoundedRect(0, 0, size.width(), size.height(), 100, 100)
        painter.drawPath(path)
        painter.end()
        result = QPixmap(size)
        result.fill(Qt.transparent)
        painter.begin(result)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return result

    def display_image(self, img, target_label=None):
        if img is not None:
            if isinstance(img, QPixmap):
                if target_label is not None:
                    target_label.setPixmap(img)
                    target_label.setScaledContents(True)
                else:
                    self.image_label.setPixmap(img)
                    self.image_label.setScaledContents(True)
            else:
                print("\033[91mUnsupported image format.\033[0m")
                if target_label is not None:
                    target_label.setText("\033[91mUnsupported image format.\033[0m")
                else:
                    self.image_label.setText("\033[91mUnsupported image format.\033[0m")
        
    def Expand_Function(self):
        if self.isFullScreen:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
            self.isFullScreen = False
            self.Expand_button.setIcon(QIcon('imgs/expand.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 70px; }")
            self.Expand_button.setToolTip("Expand")
        
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            self.isFullScreen = True
            self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Expand_button.setToolTip("Exit Full Screen (ESC)")
            

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen:
            self.Expand_Function()
        else:
            super().keyPressEvent(event)
            

    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        self.image_label.setStyleSheet("background-image: url(imgs/Conventional_CAD_Poster.png);background-image: no-repeat;")
        self.image_label2.setText("Conventional CAD Output")
        self.Features_input = []
        self.set_image(None)
        self.set_path(None)
            
    def Feature_Extraction_and_Visualization_Mode(self):
        self.on_button_click()
        self.parent.setCentralWidget(self.Feature_Extraction_and_Visualization_Screen)
        self.Feature_Extraction_and_Visualization_Screen.show()
        self.hide()

    def closeEvent(self, event):
        msgBox = CustomMessageBox()
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()
        if reply == QMessageBox.AcceptRole:
            exit(0)
        else:
            event.ignore()


    def set_image(self, img):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image = img
    
    def set_path(self, file_path):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image_path = file_path
        
        
    def carry_out(self):
        # print(self.image)
        
        self.set_image(self.image)
        self.set_path(self.file_path)
        
        # print(f"jjj {self.Feature_Extraction_and_Visualization_Screen.Conventional_image}")
        
        
        self.Feature_Extraction_and_Visualization_Screen.load_Conventional_Image()
        
        self.Features_input = [[
                               self.Feature_Extraction_and_Visualization_Screen.get_madeial_area(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Medial_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Central_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_lateral_area(),
                               self.Feature_Extraction_and_Visualization_Screen.get_central_area(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Lateral_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['ASM'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['max_probability'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['correlation'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['homogeneity'],
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_skewness(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_mean(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_stddev(),
                               self.Feature_Extraction_and_Visualization_Screen.get_ltp_entropy(),
                               self.Feature_Extraction_and_Visualization_Screen.get_ltp_features()[0],
                               self.Feature_Extraction_and_Visualization_Screen.get_ltp_features()[7],
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_variance(),
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_entropy(),
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_features()[7],
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_features()[3]
        
        ]]
        
        
        
        Features_input = self.splash_screen.scaler_Normalize_Binary.transform(self.Features_input)
        Random_Forest_Normale_Binary_Pred = self.splash_screen.Random_Forest_Normale_Binary.predict(Features_input)

        Features_input = self.splash_screen.scaler_Normalize_3_Class.transform(self.Features_input)
        Random_Forest_Normalize_3_Class_Pred = self.splash_screen.Random_Forest_Normalize_3_Class.predict(Features_input)


        # Features_input = self.splash_screen.scaler_Normalize_4_Class.transform(self.Features_input)
        # Random_Forest_Normalize_4_Classes_Pred = self.splash_screen.Random_Forest_Normalize_4_Classes.predict(Features_input)


        Features_input= self.splash_screen.scaler_Normalize_1_2_Class.transform(self.Features_input)
        SVM_Normalize_1_2_Classes_Pred = self.splash_screen.SVM_Normalize_1_2_Classes.predict(Features_input)


        Features_input= self.splash_screen.scaler_Normalize_3_4_Class.transform(self.Features_input)
        Random_Forest_Normalize_3_4Classes_Pred = self.splash_screen.Random_Forest_Normalize_3_4Classes.predict(Features_input)

        
        # Features_input= self.splash_screen.scaler_Normalize_5_Class.transform(self.Features_input)
        # Random_Forest_Normalize_5_Class_Pred = self.splash_screen.Random_Forest_Normalize_5_Class.predict(Features_input)

# _______________________________


        if Random_Forest_Normalize_3_Class_Pred == 0 & Random_Forest_Normale_Binary_Pred == 0:
            predictions = 0
        
        # else:
            # predictions = Random_Forest_Normalize_4_Classes_Pred
            
        elif Random_Forest_Normalize_3_Class_Pred == 1 :
            predictions = SVM_Normalize_1_2_Classes_Pred


        elif Random_Forest_Normalize_3_Class_Pred == 2:
            predictions = Random_Forest_Normalize_3_4Classes_Pred
            
# _______________________________
        
        
        if (predictions == 0):
            
            self.image_label2.clear()
            self.image_label2.setText(f"Class: {predictions}\n\n Normal - You are Healthy,")
            
        elif (predictions == 1):
            
            self.image_label2.clear()
            self.image_label2.setText(f"Class: {predictions}\n\n Doubtful - OsteoArthritis,")

        elif (predictions == 2):
            
            self.image_label2.clear()
            self.image_label2.setText(f"Class: {predictions}\n\n Mild - OsteoArthritis,")
            
        elif (predictions == 3):
            
            self.image_label2.clear()
            self.image_label2.setText(f"Class: {predictions}\n\n Moderate - OsteoArthritis,")
        
        elif (predictions == 4):
            
            self.image_label2.clear()
            self.image_label2.setText(f"Class: {predictions}\n\n Severe - OsteoArthritis,")
            
        else:
            self.image_label2.clear()
            
#  _____________________________________________ Histogram Window ____________________________________________________

class HistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(150, 400, 800, 400)
        self.Histogram_Intensity_label = QLabel("Intensity Histogram Window")
        self.Histogram_Intensity_label.setStyleSheet("color: red;font:25px")
        self.Histogram_Intensity_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.Histogram_Intensity_label)
        self.setLayout(layout)
        self.histogram = None

    def set_histogram(self, histogram):
        self.histogram = histogram
        self.display_histogram()

    def display_histogram(self):
        if self.histogram is not None:
            max_value = max(self.histogram)
            hist_image = np.zeros((256, 256, 3), dtype=np.uint8)

            for i in range(256):
                if max_value != 0:
                    normalized_value = int(self.histogram[i] * 255 / max_value)
                else:
                    normalized_value = 0
                color = QColor(255, 255 - normalized_value, 255 - normalized_value)
                hist_image[255 - normalized_value:, i, :] = color.getRgb()[:3]

            height, width, channel = hist_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(hist_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.Histogram_Intensity_label.setPixmap(QPixmap.fromImage(q_img))
            self.Histogram_Intensity_label.setScaledContents(True)

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close Histogram Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("Do you want to save changes before closing?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            self.display_histogram()
            pixmap = self.Histogram_Intensity_label.grab()
            pixmap.save("Outputs/histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
            
#  _____________________________________________ LBP Histogram Window ________________________________________________

class LBPHistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Binary Pattern Histogram")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(150, 150, 800, 400)
        self.LBPHistogramWindow_label = QLabel("LBP Histogram Window")
        self.LBPHistogramWindow_label.setStyleSheet("color: white;font:25px")
        self.LBPHistogramWindow_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.LBPHistogramWindow_label)
        self.setLayout(layout)

    def set_lbp_histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title("Local Binary Pattern Histogram")
            plt.xlabel("LBP Patterns")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Outputs/lbp_histogram.png")
            lbp_histogram_image = QImage("Outputs/lbp_histogram.png")
            self.LBPHistogramWindow_label.setPixmap(QPixmap.fromImage(lbp_histogram_image))
            self.LBPHistogramWindow_label.setScaledContents(True)
            plt.close()

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close LBP Histogram Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("Do you want to save changes before closing?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap = self.LBPHistogramWindow_label.grab()
            pixmap.save("Outputs/lbp_histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()

#  _____________________________________________ LTP Histogram Window ________________________________________________

class LTPHistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Ternary Pattern Histogram")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(1050, 150, 800, 400)
        self.LTPHistogramWindow_label = QLabel("LTP Histogram Window")
        self.LTPHistogramWindow_label.setStyleSheet("color: white;font:25px")
        self.LTPHistogramWindow_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.LTPHistogramWindow_label)
        self.setLayout(layout)

    def set_LTPHistogramWindow_histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title("Local Ternary Pattern Histogram")
            plt.xlabel("LTP Patterns")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Outputs/ltp_histogram.png")
            ltp_histogram_image = QImage("Outputs/ltp_histogram.png")
            self.LTPHistogramWindow_label.setPixmap(QPixmap.fromImage(ltp_histogram_image))
            self.LTPHistogramWindow_label.setScaledContents(True)
            plt.close()

            
    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close LTP Histogram Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("Do you want to save changes before closing?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap = self.LTPHistogramWindow_label.grab()
            pixmap.save("Outputs/ltp_histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
            
#  _____________________________________________ LBP and LTP Images Window ___________________________________________

class LBPandLTPImgsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LBP and LTP Images")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(1050, 300, 800, 400)
        self.LBPImgWindow_label = QLabel("LBP Image")
        self.LTPImgWindow_label = QLabel("LTP Image")
        font = QFont()
        font.setPointSize(20)
        self.LBPImgWindow_label.setFont(font)
        self.LTPImgWindow_label.setFont(font)
        self.LBPImgWindow_label.setAlignment(Qt.AlignCenter)
        self.LTPImgWindow_label.setAlignment(Qt.AlignCenter)
        self.LBPImgWindow_label.setStyleSheet("color: white;")
        self.LTPImgWindow_label.setStyleSheet("color: white;")
        layout = QHBoxLayout()
        layout.addWidget(self.LBPImgWindow_label)
        layout.addWidget(self.LTPImgWindow_label)
        self.setLayout(layout)

    def set_lbp_image(self, lbp_image):
        if lbp_image is not None:
            height, width = lbp_image.shape
            bytes_per_line = width
            q_img = QImage(lbp_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.LBPImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.LBPImgWindow_label.setScaledContents(True)

    def set_ltp_image(self, ltp_image):
        if ltp_image is not None:
            height, width = ltp_image.shape
            bytes_per_line = width
            q_img = QImage(ltp_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.LTPImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.LTPImgWindow_label.setScaledContents(True)

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close LBP and LTP Images Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("You are about to close LBP and LTP Images Window, Do you want to Save?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap1 = self.LBPImgWindow_label.grab()
            pixmap2 = self.LTPImgWindow_label.grab()
            pixmap1.save("Outputs/lbp_image.png")
            pixmap2.save("Outputs/ltp_image.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
#  ____________________________ Centeral Region Width and Tibial Width Images Window _________________________________

class CenteralRegandTibialWidthImgsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Centeral Region Width and Tibial Width Images")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(150, 300, 800, 400)
        self.CRegImgWindow_label = QLabel("Centeral Region Width Image")
        self.TibImgWindow_label = QLabel("Tibial Width Image")
        font = QFont()
        font.setPointSize(20)
        self.CRegImgWindow_label.setFont(font)
        self.TibImgWindow_label.setFont(font)
        self.CRegImgWindow_label.setAlignment(Qt.AlignCenter)
        self.TibImgWindow_label.setAlignment(Qt.AlignCenter)
        self.CRegImgWindow_label.setStyleSheet("color: white;")
        self.TibImgWindow_label.setStyleSheet("color: white;")
        layout = QHBoxLayout()
        layout.addWidget(self.CRegImgWindow_label)
        layout.addWidget(self.TibImgWindow_label)
        self.setLayout(layout)

    def set_CRegWidth_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.CRegImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.CRegImgWindow_label.setScaledContents(True)

    def set_TibialWidth_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.TibImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.TibImgWindow_label.setScaledContents(True)

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close Centeral Region Width and Tibial Width Images Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("You are about to close Centeral Region Width and Tibial Width Images Window, Do you want to Save?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap1 = self.CRegImgWindow_label.grab()
            pixmap2 = self.TibImgWindow_label.grab()
            pixmap1.save("Outputs/Centeral Region Width.png")
            pixmap2.save("Outputs/Tibial Width.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
#  ___________________________________ Predicted JSN Original Images Window _________________________________________

class PredictedJSNOriginalWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predicted JSN Original Image Window")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(1050, 300, 400, 400)
        self.Predicted_JSN_label = QLabel("Predicted JSN Original Image")
        font = QFont()
        font.setPointSize(20)
        self.Predicted_JSN_label.setFont(font)
        self.Predicted_JSN_label.setAlignment(Qt.AlignCenter)
        self.Predicted_JSN_label.setStyleSheet("color: white;")
        layout = QHBoxLayout()
        layout.addWidget(self.Predicted_JSN_label)
        self.setLayout(layout)

    def set_Predicted_JSN_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.Predicted_JSN_label.setPixmap(QPixmap.fromImage(q_img))
            self.Predicted_JSN_label.setScaledContents(True)

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close Predicted JSN Original Image Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("You are about to close Predicted JSN Original Image Window, Do you want to Save?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap1 = self.Predicted_JSN_label.grab()
            pixmap1.save("Outputs/Predicted JSN Original Image.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
            
#  ___________________________________ Histogram of Oriented Gradients Image _________________________________________

class HistogramOfOrientedGradientsImage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram of Oriented Gradients Image Window")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(1450, 300, 400, 400)
        self.HOG_image_label = QLabel("HOG Image")
        font = QFont()
        font.setPointSize(20)
        self.HOG_image_label.setFont(font)
        self.HOG_image_label.setAlignment(Qt.AlignCenter)
        self.HOG_image_label.setStyleSheet("color: white;")
        layout = QHBoxLayout()
        layout.addWidget(self.HOG_image_label)
        self.setLayout(layout)

    def set_HOG_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.HOG_image_label.setPixmap(QPixmap.fromImage(q_img))
            self.HOG_image_label.setScaledContents(True)

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close Histogram of Oriented Gradients (HOG) Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("You are about to close Histogram of Oriented Gradients (HOG) Window, Do you want to Save?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap1 = self.HOG_image_label.grab()
            pixmap1.save("Outputs/Histogram of Oriented Gradients (HOG) Window.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
            
#  _______________________________________________ HOG Histogram _____________________________________________________

class HOGHistogram(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HOG Histogram")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setGeometry(1050, 150, 800, 400)
        self.HOGHistogram_label = QLabel("HOG Histogram")
        self.HOGHistogram_label.setStyleSheet("color: white;font:25px")
        self.HOGHistogram_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.HOGHistogram_label)
        self.setLayout(layout)

    def set_HOG_Histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title(" Histogram of Oriented Gradients")
            plt.xlabel("HOG Bins")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Outputs/HOG_Histogram.png")
            HOG_Histogram_img = QImage("Outputs/HOG_Histogram.png")
            self.HOGHistogram_label.setPixmap(QPixmap.fromImage(HOG_Histogram_img))
            self.HOGHistogram_label.setScaledContents(True)
            plt.close()

            
    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close HOG Histogram Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("Do you want to save changes before closing?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap = self.HOGHistogram_label.grab()
            pixmap.save("Outputs/HOG_Histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
            
#  _____________________________________________ Feature Extraction and Visualization Screen ____________________________________________________________
class Feature_Extraction_and_Visualization_Screen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setWindowTitle("Computer aided Diagnosis")
        # self.setFixedSize(1000, 800)
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.isFullScreen = True
        self.showFullScreen()
        
        self.folder_name = "Outputs"
        folder_path = Path(self.folder_name)
        if not folder_path.is_dir():
            folder_path.mkdir()
            print(f"Folder '{self.folder_name}' created successfully.")
        else:
            print(f"Folder '{self.folder_name}' already exists.")
        
        self.splash_screen = SplashScreen(self)
        self.setCentralWidget(self.splash_screen)
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.show_main_content)
        self.loading_timer.start(1000)    

    def show_main_content(self):
        self.loading_timer.stop()
        self.splash_screen.hide()       
        self.LBPHistogramWindow = LBPHistogramWindow()
        self.LBPHistogramWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.LTPHistogramWindow = LTPHistogramWindow()
        self.LTPHistogramWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.LBPandLTPImgsWindow = LBPandLTPImgsWindow()
        self.LBPandLTPImgsWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.CenteralRegandTibialWidthImgsWindow = CenteralRegandTibialWidthImgsWindow()
        self.CenteralRegandTibialWidthImgsWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.PredictedJSNOriginalWindow = PredictedJSNOriginalWindow()
        self.PredictedJSNOriginalWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.HistogramOfOrientedGradientsImage = HistogramOfOrientedGradientsImage()
        self.HistogramOfOrientedGradientsImage.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.HOGHistogram = HOGHistogram()
        self.HOGHistogram.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        self.LBPHistogramWindow.setStyleSheet(gradient_style)
        self.LTPHistogramWindow.setStyleSheet(gradient_style)
        self.LBPandLTPImgsWindow.setStyleSheet(gradient_style)
        self.CenteralRegandTibialWidthImgsWindow.setStyleSheet(gradient_style)
        self.PredictedJSNOriginalWindow.setStyleSheet(gradient_style)
        self.HistogramOfOrientedGradientsImage.setStyleSheet(gradient_style)
        self.HOGHistogram.setStyleSheet(gradient_style)
        self.image_label = ImageLabel("Manual Feature Extractor")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMouseTracking(True)
        self.image_label.setToolTip("Double click or Drag and drop elsewhere to Add a New image")
        self.JSN_label = ImageLabel(self)
        self.JSN_label.setAlignment(Qt.AlignCenter)
        self.JSN_label.setStyleSheet("background-image: url(imgs/Feature_Extraction.png);background-image: no-repeat;")
        self.JSN_label.setScaledContents(True)
        self.JSN_label.setMouseTracking(True)
        self.JSN_label.mouseMoveEvent = self.showToolTip
        self.YOLO_label = ImageLabel("ROI Detector")
        self.YOLO_label.setAlignment(Qt.AlignCenter)
        self.YOLO_label.setScaledContents(True)
        self.YOLO_label.setMouseTracking(True)
        self.YOLO_label.mouseMoveEvent = self.showToolTip_ROI_Detector
        self.Intensity_label = ImageLabel("Intensity normalization")
        self.Intensity_label.setAlignment(Qt.AlignCenter)
        self.Intensity_label.setStyleSheet("color: white;")
        self.Binarization_label = ImageLabel("Binarization")
        self.Binarization_label.setAlignment(Qt.AlignCenter)
        self.Binarization_label.setStyleSheet("color: white;")
        self.edge_label = ImageLabel("Canny Edge Detection")
        self.edge_label.setAlignment(Qt.AlignCenter)
        self.edge_label.setStyleSheet("color: white;")
        font = QFont()
        font.setPointSize(20)
        self.image_label.setFont(font)
        self.JSN_label.setFont(font)
        self.YOLO_label.setFont(font)
        self.Intensity_label.setFont(font)
        self.Binarization_label.setFont(font)
        self.edge_label.setFont(font)            
        layout = QVBoxLayout()
        layoutH1 = QHBoxLayout()
        layout.addLayout(layoutH1)
        layoutH1.addWidget(self.image_label)
        layoutH1.addWidget(self.JSN_label)
        layoutH1.addWidget(self.YOLO_label)
        layoutH = QHBoxLayout()
        layout.addLayout(layoutH)
        layoutH2 = QHBoxLayout()
        layout.addLayout(layoutH2)
        self.normalize_checkbox = QCheckBox("Enable Normalizing Mode", self)
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.setStyleSheet("color: white;font-size: 16px;")
        self.normalize_checkbox.stateChanged.connect(self.toggle_intensity_normalization)
        self.automatic_label = ImageLabel("")
        self.automatic_label.setFixedSize(850,50)
        self.automatic_label.setStyleSheet("color: grey; background-color: black;")
        self.automatic_label.setAlignment(Qt.AlignCenter)
        self.automatic_label.setScaledContents(True)
        font = QFont()
        font.setPointSize(15)
        self.automatic_label.setFont(font)
        self.Auto_F_Extractor = QPushButton(self)
        self.Auto_F_Extractor.setIcon(QIcon('imgs/enable-mode.png'))
        self.Auto_F_Extractor.setMinimumSize(40,40)
        self.Auto_F_Extractor.setMaximumSize(40,40)
        self.Auto_F_Extractor.setIconSize(QSize(40,40))
        self.Auto_F_Extractor.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Auto_F_Extractor.clicked.connect(self.switch_mode)
        self.Auto_F_Extractor.setToolTip("switch to Data set Automatic Feature Extractor")
        self.home_button = QPushButton(self)
        homeicon = QIcon("imgs/clear.png")
        self.home_button.setIcon(homeicon)
        icon_size = QSize(50, 50)
        self.home_button.setIconSize(icon_size)
        self.home_button.setStyleSheet("background-color: transparent; border: none;")
        self.home_button.clicked.connect(self.on_button_click)
        self.home_button.setToolTip("Clear")
        self.Windows_button = QPushButton(self)
        self.Windows_button.setIcon(QIcon('imgs/hide.png'))
        self.Windows_button.setMinimumSize(35,35)
        self.Windows_button.setMaximumSize(35,35)
        self.Windows_button.setIconSize(QSize(35,35))
        self.Windows_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Windows_button.clicked.connect(self.Windows_button_Toggling)
        self.Windows_button.setToolTip("Show")
        self.Expand_button = QPushButton(self)
        self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
        self.Expand_button.setMinimumSize(50,50)
        self.Expand_button.setMaximumSize(50,50)
        self.Expand_button.setIconSize(QSize(50,50))
        self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Expand_button.clicked.connect(self.Expand_Function)
        self.Expand_button.setToolTip("Exit Full Screen (ESC)")
        self.Restart_button = QPushButton(self)
        self.Restart_button.setIcon(QIcon('imgs/restart.png'))
        self.Restart_button.setMinimumSize(35,35)
        self.Restart_button.setMaximumSize(35,35)
        self.Restart_button.setIconSize(QSize(35,35))
        self.Restart_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Restart_button.clicked.connect(self.restart_code)
        self.Restart_button.setToolTip("Restart")
        self.EXIT_button = QPushButton(self)
        self.EXIT_button.setIcon(QIcon('imgs/exit.png'))
        self.EXIT_button.setMinimumSize(60,60)
        self.EXIT_button.setMaximumSize(60,60)
        self.EXIT_button.setIconSize(QSize(60,60))
        self.EXIT_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.EXIT_button.clicked.connect(self.EXIT_Function)
        self.EXIT_button.setToolTip("EXIT")
        self.Ai_Automated_CAD_button = QPushButton(self)
        self.Ai_Automated_CAD_button.setIcon(QIcon('imgs/Ai_Automated_CAD_ICON.png'))
        self.Ai_Automated_CAD_button.setMinimumSize(95,95)
        self.Ai_Automated_CAD_button.setMaximumSize(95,95)
        self.Ai_Automated_CAD_button.setIconSize(QSize(95,95))
        self.Ai_Automated_CAD_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Ai_Automated_CAD_button.clicked.connect(self.AI_Automated_CAD)
        self.Ai_Automated_CAD_button.setToolTip("Redirect to Ai Automated CAD")       
        self.Conventional_CAD_button = QPushButton(self)
        self.Conventional_CAD_button.setIcon(QIcon('imgs/Conventional_CAD_Poster.png'))
        self.Conventional_CAD_button.setMinimumSize(80,80)
        self.Conventional_CAD_button.setMaximumSize(80,80)
        self.Conventional_CAD_button.setIconSize(QSize(80,80))
        self.Conventional_CAD_button.setStyleSheet("QPushButton { border-radius: 50px; }")
        self.Conventional_CAD_button.clicked.connect(self.Conventional_CAD)
        self.Conventional_CAD_button.setToolTip("Redirect to Conventional CAD")
        layout.addLayout(layoutH2)
        layoutH2.addWidget(self.normalize_checkbox)
        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layoutH2.addItem(spacer)  
        layoutH2.addWidget(self.automatic_label)  
        layoutH2.addWidget(self.Auto_F_Extractor)
        layoutH2.addWidget(self.Windows_button)
        layoutH2.addWidget(self.Expand_button)
        layoutH2.addWidget(self.home_button)   
        layoutH2.addWidget(self.Restart_button)
        layoutH2.addWidget(self.EXIT_button)
        layoutH2.addWidget(self.Conventional_CAD_button)
        layoutH2.addWidget(self.Ai_Automated_CAD_button)
        layoutH.addWidget(self.Intensity_label)
        layoutH.addWidget(self.Binarization_label)
        layoutH.addWidget(self.edge_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.histogram_window = HistogramWindow()
        self.histogram_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.histogram_window.show()
        self.histogram_window.set_histogram(None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_histogram_display)
        self.timer.start(1)
        # _____________________________________________ Iitialization _______________________________________________
        self.switch = True
        self.switch2 = False
        self.mode = True        
        self.image = None
        self.image_path = None
        self.Conventional_image = None
        self.Conventional_image_path = None
        self.cropped_images = []
        self.user_cropped = False        
        self.setAcceptDrops(True)
        self.switch2_toggle()
        self.center_crop = ()
        self.perform_intensity_normalization = True
        self.perform_canny_edge = True
        self.end_Automation = 0
        self.conventional_image_Indicator = 0
        self.check_knee_literality = False
        self.prob = 0.0
        self.Centeral_Left_X = 0
        self.Centeral_Right_X = 0
        self.Centeral_Left_Y  = 0
        self.Centeral_Right_Y = 0
        self.Centeral_Region_Width = 0
        self.Centeral_Region_Height = 0
        self.Centeral_Region_Center = ()
        self.Xc =  0 
        self.Yc = 0
        self.Tibial_width = 0
        self.start_Tibial_width = 0
        self.end_Tibial_width = 0
        self.average_medial_distance = 0
        self.average_central_distance = 0
        self.average_lateral_distance = 0
        self.average_medial_distance_mm = 0
        self.average_central_distance_mm = 0
        self.average_lateral_distance_mm = 0
        self.Medial_ratio = 0
        self.Central_ratio = 0
        self.Lateral_ratio = 0
        self.average_distance = 0
        self.average_distance_mm = 0
        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0
        self.medial_area = 0
        self.central_area = 0
        self.lateral_area = 0
        self.medial_area_Squaredmm = 0
        self.central_area_Squaredmm = 0
        self.lateral_area_Squaredmm = 0
        self.medial_area_Ratio = 0
        self.central_area_Ratio = 0
        self.lateral_area_Ratio = 0
        self.JSN_Area_Total = 0
        self.JSN_Area_Total_Squared_mm = 0
        self.Tibial_width_Predicted_Area = 0 
        self.Tibial_width_Predicted_Area_mm = 0
        self.medial_area_Ratio_TWPA = 0
        self.central_area_Ratio_TWPA = 0
        self.lateral_area_Ratio_TWPA = 0
        self.intensity_mean = 0
        self.intensity_stddev = 0
        self.intensity_skewness = 0
        self.intensity_kurtosis = 0
        self.cooccurrence_properties = {}
        self.lbp_features = []
        self.lbp_variance = 0
        self.lbp_entropy = 0
        self.ltp_features = []
        self.ltp_variance = 0
        self.ltp_entropy = 0
        self.hog_bins = []
        #                        ____________________ Calculate Area Ratio______________________________
        self.screw_thickness1 = 4.5
        self.screw_thickness2 = 3.0
        self.screw_length = 35
        self.screw_area_standard = self.screw_thickness1* 0.75* self.screw_length + self.screw_thickness2* 0.25* self.screw_length
        self.screw_area_virtual = 471 #Squared pixel
        self.area_ratio = self.screw_area_virtual /  self.screw_area_standard
        #                        ____________________ Calculate Length Ratio______________________________
        self.virtual_length = 140 #unit pixel
        self.length_ratio = self.virtual_length / self.screw_length

#  __________________________________________________ Functions _______________________________________________________    
        
    def showToolTip(self, event):
        if self.image is not None:            
            if (self.check_knee_literality):
                subregions_area = ["Medial Area (Squared mm)", "Central Area (Squared mm)", "Lateral Area (Squared mm)"]
                subregions_distance = ["Medial Distance (mm)", "Central Distance (mm)", "Lateral Distance (mm)"]
                areas = [self.medial_area_Squaredmm, self.central_area_Squaredmm, self.lateral_area_Squaredmm]
                distances = [self.average_medial_distance_mm, self.average_central_distance_mm, self.average_lateral_distance_mm]
            else:
                subregions_area = ["Lateral Area (Squared mm)", "Central Area (Squared mm)", "Medial Area (Squared mm)"]
                subregions_distance = ["Lateral Distance (mm)", "Central Distance (mm)", "Medial Distance (mm)"]
                areas = [self.lateral_area_Squaredmm, self.central_area_Squaredmm, self.medial_area_Squaredmm]
                distances = [self.average_lateral_distance_mm, self.average_central_distance_mm, self.average_medial_distance_mm]
                
            subregion_width = self.JSN_label.width() / 3
            subregion_boundaries = [(i * subregion_width, (i + 1) * subregion_width) for i in range(3)]
            for i, (start, end) in enumerate(subregion_boundaries):
                if start <= event.pos().x() < end:
                    pos = self.mapToGlobal(self.JSN_label.pos())
                    rect = self.JSN_label.rect()
                    QToolTip.showText(event.globalPos(), f"{subregions_area[i]}: {areas[i]}\n{subregions_distance[i]}: {distances[i]} ", self, QRect(pos.x(), pos.y(), rect.width(), rect.height()))
                    return

            QToolTip.hideText()    
            
        else:
            pass
            
        
    
    def showToolTip_ROI_Detector(self, event):
        if self.YOLO_label.text() == "ROI Detector":
            pass
        else:
            if self.prob is not None:
                QToolTip.showText(event.globalPos(), f"Probability is {self.prob}")
            else:
                QToolTip.showText(event.globalPos(), "")


        
    def AI_Automated_CAD2(self, event):
        # self.on_button_click()
        self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(parent = self)
        self.AI_Automated_CAD_Screen.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.AI_Automated_CAD_Screen.isFullScreen = True
        self.AI_Automated_CAD_Screen.showFullScreen()
        self.setCentralWidget(self.AI_Automated_CAD_Screen)
        self.AI_Automated_CAD_Screen.show()
        
        
        
    def AI_Automated_CAD(self, event):
        self.on_button_click()
        self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(parent = self)
        self.AI_Automated_CAD_Screen.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.AI_Automated_CAD_Screen.isFullScreen = True
        self.AI_Automated_CAD_Screen.showFullScreen()
        self.setCentralWidget(self.AI_Automated_CAD_Screen)
        self.AI_Automated_CAD_Screen.show()
        
        
    def Conventional_CAD(self, event):
        self.on_button_click()
        self.Conventional_CAD_Screen = Conventional_CAD_Screen(parent = self)
        self.Conventional_CAD_Screen.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.Conventional_CAD_Screen.isFullScreen = True
        self.Conventional_CAD_Screen.showFullScreen()
        self.setCentralWidget(self.Conventional_CAD_Screen)
        self.Conventional_CAD_Screen.show()
            
    def Expand_Function(self):
        if self.isFullScreen:
            self.setWindowFlags(Qt.Window)
            self.showNormal()
            self.isFullScreen = False
            self.Expand_button.setIcon(QIcon('imgs/expand.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 70px; }")
            self.Expand_button.setToolTip("Expand")
        
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            self.isFullScreen = True
            self.Expand_button.setIcon(QIcon('imgs/reduce.png'))
            self.Expand_button.setMinimumSize(50,50)
            self.Expand_button.setMaximumSize(50,50)
            self.Expand_button.setIconSize(QSize(50,50))
            self.Expand_button.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Expand_button.setToolTip("Exit Full Screen (ESC)")
            

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen:
            self.Expand_Function()
        else:
            super().keyPressEvent(event)
            
   
    def restart_code(self):
        QApplication.quit()
        subprocess.Popen([sys.executable] + sys.argv)
        
        
    def EXIT_Function(self, event):
        self.closeEvent(event)
        
        
    def closeEvent(self, event):
        msgBox = CustomMessageBox()
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            exit(0)
        else:
            pass          
                
    def switch_mode(self):
        if (self.switch == True):
            self.Auto_F_Extractor.setIcon(QIcon('imgs/disable-mode.png'))
            self.Auto_F_Extractor.setMinimumSize(40,40)
            self.Auto_F_Extractor.setMaximumSize(40,40)
            self.Auto_F_Extractor.setIconSize(QSize(40,40))
            self.Auto_F_Extractor.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Auto_F_Extractor.setToolTip("switch to An image Automatic Feature Extractor")
            text1 = "Hint: ESC FullScreen mode before starting automation.......\n"
            text2 = "Don't panic if it looks stuck......"
            text = text1 + text2
            self.automatic_label.setText(text)   
            self.switch= False
        else:
            self.Auto_F_Extractor.setIcon(QIcon('imgs/enable-mode.png'))
            self.Auto_F_Extractor.setMinimumSize(40,40)
            self.Auto_F_Extractor.setMaximumSize(40,40)
            self.Auto_F_Extractor.setIconSize(QSize(40,40))
            self.Auto_F_Extractor.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Auto_F_Extractor.setToolTip("switch to Data set Automatic Feature Extractor")
            self.automatic_label.setText("")    
            self.switch= True


    def Windows_button_Toggling(self):
        if (self.switch2 == False):
            self.Windows_button.setIcon(QIcon('imgs/visible.png'))
            self.Windows_button.setMinimumSize(35,35)
            self.Windows_button.setMaximumSize(35,35)
            self.Windows_button.setIconSize(QSize(35,35))
            self.Windows_button.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Windows_button.setToolTip("Hide")
            self.switch2 = True
            self.switch2_toggle()
        else:
            self.Windows_button.setIcon(QIcon('imgs/hide.png'))
            self.Windows_button.setMinimumSize(35,35)
            self.Windows_button.setMaximumSize(35,35)
            self.Windows_button.setIconSize(QSize(35,35))
            self.Windows_button.setStyleSheet("QPushButton { border-radius: 50px; }")
            self.Windows_button.setToolTip("Show")
            self.switch2 = False
            self.switch2_toggle()
        
    def switch2_toggle(self):
        if(self.switch2):
            self.histogram_window.show()
            self.LBPHistogramWindow.show()
            self.LTPHistogramWindow.show()
            self.LBPandLTPImgsWindow.show()
            self.CenteralRegandTibialWidthImgsWindow.show()
            self.PredictedJSNOriginalWindow.show()
            self.HistogramOfOrientedGradientsImage.show()
            self.HOGHistogram.show()
            
        else:
            self.histogram_window.hide()
            self.LBPHistogramWindow.hide()
            self.LTPHistogramWindow.hide()
            self.LBPandLTPImgsWindow.hide()
            self.CenteralRegandTibialWidthImgsWindow.hide()
            self.PredictedJSNOriginalWindow.hide()            
            self.HistogramOfOrientedGradientsImage.hide()
            self.HOGHistogram.hide()

    def load_Conventional_Image(self):
        
        self.conventional_image_Indicator = 2
        self.Conventional_image = self.equalize_Original_image(self.Conventional_image)
        
        y = int(self.Conventional_image.shape[0] * 0.5)
        x = int(self.Conventional_image.shape[1] * 0.5)
        self.center_crop = (x, y)
        crop_size = 224
        x1 = x - crop_size // 2
        x2 = x + crop_size // 2
        y1 = y - crop_size // 2
        y2 = y + crop_size // 2
        x1 = max(x1, 0)
        x2 = min(x2, self.Conventional_image.shape[1])
        y1 = max(y1, 0)
        y2 = min(y2, self.Conventional_image.shape[0])
        self.Conventional_image = self.Conventional_image[y1:y2, x1:x2]
        
        
        self.display_image(self.Conventional_image, target_label=self.image_label)
        self.display_image(self.Conventional_image, target_label=self.YOLO_label)
        self.display_image(self.Conventional_image, target_label=self.JSN_label)
        self.display_image(self.Conventional_image, target_label=self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label)
        self.display_image(self.Conventional_image, target_label=self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label)
        self.display_image(self.Conventional_image, target_label=self.PredictedJSNOriginalWindow.Predicted_JSN_label)
        
        
        self.check_knee_literality = self.set_knee_literality(self.Conventional_image)
            
        self.YOLO_CReg_predict(self.Conventional_image)
        self.YOLO_TibialW_predict(self.Conventional_image)

        self.predicted_polygon = self.YOLO_predict_Segmented(self.Conventional_image)
        self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)                
        self.draw_predicted_polygon_Original(self.predicted_polygon, self.PredictedJSNOriginalWindow.Predicted_JSN_label)
        
        
        self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

        if (self.check_knee_literality):
            self.medial_area = self.red_area
            self.central_area = self.green_area
            self.lateral_area = self.blue_area
            
        else:
            self.medial_area = self.blue_area
            self.central_area = self.green_area
            self.lateral_area = self.red_area
                            
        
        print(f"\033[92mMedial Area = {self.medial_area}.Squared Pixel\033[0m")
        print(f"\033[92mCentral Area = {self.central_area}.Squared Pixel\033[0m")
        print(f"\033[92mLateral Area = {self.lateral_area}.Squared Pixel\033[0m")
        
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
        
        self.medial_area_Squaredmm = self.medial_area / self.area_ratio
        self.central_area_Squaredmm = self.central_area / self.area_ratio
        self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
        
        print(f"\033[92mMedial Area = {self.medial_area_Squaredmm}.Squared mm\033[0m")
        print(f"\033[92mCentral Area = {self.central_area_Squaredmm}.Squared mm\033[0m")
        print(f"\033[92mLateral Area = {self.lateral_area_Squaredmm}.Squared mm\033[0m")
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
        self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
        self.central_area_Ratio = self.central_area / self.JSN_Area_Total
        self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
        
        print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
        print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
        print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

        self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
        self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
        self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
        
        self.medial_area_Ratio_TWPA *= 100 
        self.central_area_Ratio_TWPA *= 100
        self.lateral_area_Ratio_TWPA *= 100
        
        print(f"\033[92mMedial Area Ratio TWPA (%) = {self.medial_area_Ratio_TWPA}.\033[0m")
        print(f"\033[92mCentral Area Ratio TWPA (%) = {self.central_area_Ratio_TWPA}.\033[0m")
        print(f"\033[92mLateral Area Ratio TWPA (%) = {self.lateral_area_Ratio_TWPA}.\033[0m")
    
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")






        yolo_results = self.YOLO_predict(self.Conventional_image)
        self.calculate_and_save_features_YOLO(yolo_results, self.Conventional_image_path,self.Conventional_image)
        print(f"\033[92myolo_results = {yolo_results}.\033[0m")
        self.save_PNGs()
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        self.on_button_click()
        

    def load_image(self, image_path):
        self.user_cropped = False
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        y = int(self.image.shape[0] * 0.5)
        x = int(self.image.shape[1] * 0.5)
        self.center_crop = (x, y)
        crop_size = 224
        x1 = x - crop_size // 2
        x2 = x + crop_size // 2
        y1 = y - crop_size // 2
        y2 = y + crop_size // 2
        x1 = max(x1, 0)
        x2 = min(x2, self.image.shape[1])
        y1 = max(y1, 0)
        y2 = min(y2, self.image.shape[0])
        self.image = self.image[y1:y2, x1:x2]

        if (self.perform_intensity_normalization):
            self.image = self.equalize_Original_image(self.image)
        else:
            pass
        
        self.display_image(self.image, target_label=self.image_label)
        self.display_image(self.image, target_label=self.YOLO_label)
        self.display_image(self.image, target_label=self.JSN_label)
        self.display_image(self.image, target_label=self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label)
        self.display_image(self.image, target_label=self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label)
        self.display_image(self.image, target_label=self.PredictedJSNOriginalWindow.Predicted_JSN_label)
          

        if (self.switch):
            directory_path = os.path.dirname(image_path)            
            self.folder_name = os.path.basename(directory_path)
            
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            self.check_knee_literality = self.set_knee_literality(self.image)
            
            self.YOLO_CReg_predict(self.image)
            self.YOLO_TibialW_predict(self.image)

            self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
            self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)                
            self.draw_predicted_polygon_Original(self.predicted_polygon, self.PredictedJSNOriginalWindow.Predicted_JSN_label)

            self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

            if (self.check_knee_literality):
                self.medial_area = self.red_area
                self.central_area = self.green_area
                self.lateral_area = self.blue_area
                
            else:
                self.medial_area = self.blue_area
                self.central_area = self.green_area
                self.lateral_area = self.red_area
                                
            print(f"\033[92mMedial Area = {self.medial_area}.Squared Pixel\033[0m")
            print(f"\033[92mCentral Area = {self.central_area}.Squared Pixel\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area}.Squared Pixel\033[0m")
            
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            
            self.medial_area_Squaredmm = self.medial_area / self.area_ratio
            self.central_area_Squaredmm = self.central_area / self.area_ratio
            self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
            
            print(f"\033[92mMedial Area = {self.medial_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mCentral Area = {self.central_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
            self.central_area_Ratio = self.central_area / self.JSN_Area_Total
            self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
            
            print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
            print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
            print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
            
            
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
            self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
            self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
            self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
            
            
            self.medial_area_Ratio_TWPA *= 100 
            self.central_area_Ratio_TWPA *= 100
            self.lateral_area_Ratio_TWPA *= 100
        
        
            print(f"\033[92mMedial Area Ratio TWPA (%) = {self.medial_area_Ratio_TWPA}.\033[0m")
            print(f"\033[92mCentral Area Ratio TWPA (%) = {self.central_area_Ratio_TWPA}.\033[0m")
            print(f"\033[92mLateral Area Ratio TWPA (%) = {self.lateral_area_Ratio_TWPA}.\033[0m")
            
        
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            yolo_results = self.YOLO_predict(self.image)
            self.calculate_and_save_features_YOLO(yolo_results, image_path, self.image)
            print(f"\033[92myolo_results = {yolo_results}.\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

        else:
            self.current = image_path
            directory_path = os.path.dirname(image_path)
            image_files = glob.glob(os.path.join(directory_path, "*.png"))
            self.folder_name = os.path.basename(directory_path)
            
            counter = 0
            start_time = time.time()
            for image_path in image_files:
                counter+=1
                self.image_path = image_path
                self.image = cv2.imread(self.image_path)
                if (self.perform_intensity_normalization):
                    self.image = self.equalize_Original_image(self.image)
                else:
                    pass
                
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
                self.check_knee_literality = self.set_knee_literality(self.image)
                
                self.YOLO_CReg_predict(self.image)
                self.YOLO_TibialW_predict(self.image)
                
                self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
                self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)
                self.draw_predicted_polygon_Original(self.predicted_polygon, self.PredictedJSNOriginalWindow.Predicted_JSN_label)

                self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

                if (self.check_knee_literality):
                    self.medial_area = self.red_area
                    self.central_area = self.green_area
                    self.lateral_area = self.blue_area
                    
                else:
                    self.medial_area = self.blue_area
                    self.central_area = self.green_area
                    self.lateral_area = self.red_area
                                    
                print(f"\033[92mMedial Area = {self.medial_area}.Squared Pixel\033[0m")
                print(f"\033[92mCentral Area = {self.central_area}.Squared Pixel\033[0m")
                print(f"\033[92mLateral Area = {self.lateral_area}.Squared Pixel\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

                self.medial_area_Squaredmm = self.medial_area / self.area_ratio
                self.central_area_Squaredmm = self.central_area / self.area_ratio
                self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
            
                print(f"\033[92mMedial Area = {self.medial_area_Squaredmm}.Squared mm\033[0m")
                print(f"\033[92mCentral Area = {self.central_area_Squaredmm}.Squared mm\033[0m")
                print(f"\033[92mLateral Area = {self.lateral_area_Squaredmm}.Squared mm\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            
            
                self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total

                print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
                print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
                print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            
                self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
                self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
                self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
                
                self.medial_area_Ratio_TWPA *= 100 
                self.central_area_Ratio_TWPA *= 100
                self.lateral_area_Ratio_TWPA *= 100
        
        
                print(f"\033[92mMedial Area Ratio TWPA (%) = {self.medial_area_Ratio_TWPA}.\033[0m")
                print(f"\033[92mCentral Area Ratio TWPA (%) = {self.central_area_Ratio_TWPA}.\033[0m")
                print(f"\033[92mLateral Area Ratio TWPA (%) = {self.lateral_area_Ratio_TWPA}.\033[0m")
            
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

                yolo_results = self.YOLO_predict(self.image)
                self.calculate_and_save_features_YOLO(yolo_results, self.image_path, self.image)
                print(f"\033[92mCounter = {counter}.\033[0m")
                # print(f"\033[92myolo_results = {yolo_results}.\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            end_time = time.time()
            self.overall_time = end_time - start_time
            hours = self.overall_time // 3600
            remaining_time = self.overall_time % 3600
            minutes = remaining_time // 60
            seconds = remaining_time % 60

            print(f"\033[92mTotal Computational Time = {hours} Hours, {minutes} Minutes, and {seconds} Seconds.\033[0m")
            self.end_Automation = 2

            self.on_button_click()
            self.image = cv2.imread(self.current)
            if (self.perform_intensity_normalization):
                self.image = self.equalize_Original_image(self.image)
            else:
                pass
        
            self.display_image(self.image, target_label=self.image_label)
            self.display_image(self.image, target_label=self.YOLO_label)
            self.display_image(self.image, target_label=self.JSN_label)
            self.display_image(self.image, target_label=self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label)
            self.display_image(self.image, target_label=self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label)
            self.display_image(self.image, target_label=self.PredictedJSNOriginalWindow.Predicted_JSN_label)


            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            self.check_knee_literality = self.set_knee_literality(self.image)

            self.YOLO_CReg_predict(self.image)
            self.YOLO_TibialW_predict(self.image)
            
            self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
            self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)
            self.draw_predicted_polygon_Original(self.predicted_polygon, self.PredictedJSNOriginalWindow.Predicted_JSN_label)

            self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

            if (self.check_knee_literality):
                self.medial_area = self.red_area
                self.central_area = self.green_area
                self.lateral_area = self.blue_area
                
            else:
                self.medial_area = self.blue_area
                self.central_area = self.green_area
                self.lateral_area = self.red_area
                
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
                                
            print(f"\033[92mMedial Area = {self.medial_area}.Squared Pixel\033[0m")
            print(f"\033[92mCentral Area = {self.central_area}.Squared Pixel\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area}.Squared Pixel\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")        
            
            self.medial_area_Squaredmm = self.medial_area / self.area_ratio
            self.central_area_Squaredmm = self.central_area / self.area_ratio
            self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
                
            print(f"\033[92mMedial Area = {self.medial_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mCentral Area = {self.central_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area_Squaredmm}.Squared mm\033[0m")

            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
            self.central_area_Ratio = self.central_area / self.JSN_Area_Total
            self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
            
            print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
            print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
            print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
            self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
            self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
            self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
            
            self.medial_area_Ratio_TWPA *= 100 
            self.central_area_Ratio_TWPA *= 100
            self.lateral_area_Ratio_TWPA *= 100
        
            print(f"\033[92mMedial Area Ratio TWPA (%) = {self.medial_area_Ratio_TWPA}.\033[0m")
            print(f"\033[92mCentral Area Ratio TWPA (%) = {self.central_area_Ratio_TWPA}.\033[0m")
            print(f"\033[92mLateral Area Ratio TWPA (%) = {self.lateral_area_Ratio_TWPA}.\033[0m")
        
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")




            yolo_results = self.YOLO_predict(self.image)
            print(f"\033[92myolo_results = {yolo_results}.\033[0m")
            # self.print_to_cmd(f"\033[92myolo_results = {yolo_results}.\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

        if self.perform_intensity_normalization:
            self.equalize_histogram()
            self.binarization()

        if self.perform_canny_edge:
            self.save_PNGs()
            img = cv2.imread("Outputs/JSN-Region.png")
            self.perform_edge_detection(img)

        self.calculate_histogram()
        self.save_PNGs()


    def print_to_cmd(self, message):
        subprocess.Popen(["start", "cmd", "/k", "echo", message], shell=True)


    def equalize_Original_image(self, img):
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(original_gray)
        self.equalized_image_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
        
        return self.equalized_image_rgb
    

    def display_image(self, img, target_label=None):
        if img is not None:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    height, width = img.shape
                    bytes_per_line = width
                    format = QImage.Format_Grayscale8
                else:
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    format = QImage.Format_RGB888
                q_img = QImage(img.data, width, height, bytes_per_line, format)
            elif isinstance(img, QImage):
                q_img = img
            else:
                print("\033[91mUnsupported image format.\033[0m")
                return
            if target_label is not None:
                target_label.setPixmap(QPixmap.fromImage(q_img))
                target_label.setScaledContents(True)
            else:
                self.image_label.setPixmap(QPixmap.fromImage(q_img))
                self.image_label.setScaledContents(True)



    def set_knee_literality(self, img):
        resize = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        device = torch.device("cpu")
        resize_tensor = torch.from_numpy(np.expand_dims(np.transpose(resize, (2, 0, 1)), axis=0)).to(device)
        output = self.splash_screen.knee_Literality_model(resize_tensor)
        if output.item() > 0.5:
            print(f"\033[92mLeft Knee.\033[0m")
            return True
        else:
            print(f"\033[92mRight Knee.\033[0m")
            return False
  
    def YOLO_predict_Segmented(self, img):
        if img is not None and self.splash_screen.model_instance_Segmentation is not None:
            output = self.splash_screen.model_instance_Segmentation.predict(img,
                                                                            save=False,
                                                                            show=False,
                                                                            show_labels=False,
                                                                            show_conf=True,
                                                                            save_txt=False)
            masks = output[0].masks

            if masks is not None:
                masks = masks.xy
                self.predicted_polygon = [mask.tolist() for mask in masks]
                return self.predicted_polygon
            else:
                print("\033[91mMasks are None. No segmentation data available.\033[0m")
                return []
        else:
            print("\033[91mImage or model not available.\033[0m")
            return []

            
    def draw_predicted_polygon_Original(self, polygon, target_label):
        if polygon and target_label is not None:
            pixmap = target_label.pixmap()
            painter = QPainter(pixmap)
            color = QColor(0, 0, 0)
            # color = QColor(50, 255, 50)
            color.setAlpha(255)
            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            scaled_polygon = [QPointF(float(x), float(y)) for point in polygon for x, y in point]
            painter.drawPolygon(QPolygonF(scaled_polygon))

    def draw_predicted_polygon(self, polygon, target_label):
        if polygon and target_label is not None:
            pixmap = target_label.pixmap()
            painter = QPainter(pixmap)
            color = QColor(50, 255, 50)
            color.setAlpha(255)
            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            path = QPainterPath()
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))
            painter.setClipPath(path)
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))

            if len(polygon) > 0:
                left_x = min(point[0] for point in polygon[0])
                right_x = max(point[0] for point in polygon[0])

                medial_region_start = left_x
                medial_region_end = (self.Tibial_width / 2 - 0.5 * self.Centeral_Region_Width) + self.start_Tibial_width
                central_region_start = medial_region_end
                central_region_end = (self.Tibial_width / 2 + 0.5 * self.Centeral_Region_Width) + self.start_Tibial_width
                lateral_region_start = central_region_end
                lateral_region_end = right_x
            
                self.color_red = QColor(255, 0, 0)
                self.color_green = QColor(0, 255, 0)
                self.color_blue = QColor(0, 0, 255)
                self.color_red.setAlpha(255)
                self.color_green.setAlpha(255)
                self.color_blue.setAlpha(255)
                self.draw_subregion(painter, pixmap, polygon, medial_region_start, medial_region_end, self.color_red)
                self.draw_subregion(painter, pixmap, polygon, central_region_start, central_region_end, self.color_green)
                self.draw_subregion(painter, pixmap, polygon, lateral_region_start, lateral_region_end, self.color_blue)
                
                polygon = [[[int(coord) for coord in point] for point in sublist] for sublist in polygon]
                left_x = min(point[0] for  point in polygon[0])
                right_x = max(point[0] for point in polygon[0])
                               
                self.percentage = (right_x - left_x)/self.Tibial_width
                self.Automatic_JSW_tibial_ratio = 0.926155
                self.JSW_width = self.Automatic_JSW_tibial_ratio * self.Tibial_width
                assumed_left = self.start_Tibial_width + 0.5 * self.Tibial_width - 0.5 * self.JSW_width
                assumed_right = assumed_left + self.JSW_width
                
                polygon = [[[int(coord) for coord in point] for point in sublist] for sublist in polygon]
                two_dim_array = np.array([point for sublist in polygon for point in sublist])

                if self.end_Automation == 2:
                    self.image_path = self.current
                    self.end_Automation = 0
                else:
                    pass
                
                
                if self.conventional_image_Indicator == 2:
                    img = cv2.imread(self.Conventional_image_path)
                    self.conventional_image_Indicator = 0                 
                else:
                    img = cv2.imread(self.image_path)
                    
                self.fill_colored_polygon(img, two_dim_array)
                edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255)
                
                # if assumed_left > left_x:
                #     assumed_left = left_x
                    
                # if assumed_right < right_x:
                #     assumed_right = right_x
                  
# ______________________________________  Start Vertical Distance For Right Region  ________________________________ #

                column_index = int(self.Xc)
                row_index  = int (self.Yc) 

                Upper_index= row_index
                down_index = row_index

                distace_Central = [] 
                distace_Right = []

           
                for move_right in range (int(assumed_right - column_index)):
                    if  ((column_index +move_right)>right_x)  and((column_index +move_right)< assumed_right):
                        distace_Right.append(0)

                    else :
                        try: 
                            if edges[row_index] [column_index+ move_right] > 0  :
                                row_index = int((Upper_index + down_index)/2)

                            upper_point_flag = 0
                            upper_point = 0 
                            Upper_index = row_index
                            down_point_flag = 0
                            down_point = 0
                            
                            down_index = row_index

                        #  get upper point to get vertical distance 
                            while upper_point_flag == 0 :
                                if  edges[Upper_index] [column_index +move_right] >0:
                                    upper_point_flag = 1
                                    upper_point = Upper_index   
                                Upper_index -= 1

                        #  get dwon point to get vertical distance  
                            while down_point_flag == 0:
                                if  edges[down_index] [column_index +move_right] >0:
                                    down_point_flag = 1
                                    down_point = down_index
                                down_index += 1


                            if ((column_index +move_right) >= lateral_region_start) and ((column_index +move_right)<=right_x):
                                distace_Right.append(down_point -upper_point)

                            elif  ((column_index +move_right) < lateral_region_start) and ((column_index + move_right) >  medial_region_end):
                                distace_Central.append(down_point -upper_point)
                        
                            else:
                                pass
                        except:
                            continue 



# ___________________________________  Start Vertical Distance For Left Region  ____________________________________ #


                column_index = int(self.Xc)
                row_index  = int (self.Yc)
                
                Upper_index= row_index
                down_index = row_index

                distace_Left = []

                for move_left in range (int(column_index - assumed_left)):
                    if  ((column_index - move_left) < left_x)  and((column_index - move_left)> assumed_left):
                        distace_Left.append(0)

                    else : 
                        try:
                            if edges[row_index] [column_index- move_left] >0  :
                                row_index = int((Upper_index +down_index)/2)

                            upper_point_flag = 0
                            upper_point = 0
                            Upper_index= row_index

                            down_point_flag = 0
                            down_point = 0
                            down_index = row_index

                            # get upper point to get vertical distance 
                            while upper_point_flag == 0 :
                                if  edges[Upper_index] [column_index  - move_left] >0:
                                    upper_point_flag = 1
                                    upper_point = Upper_index   
                                Upper_index -=1

                            # get dwon point to get vertical distance 
                            while down_point_flag == 0:
                                if  edges[down_index] [column_index  - move_left] >0:
                                    down_point_flag = 1
                                    down_point = down_index
                                down_index += 1

                            if ((column_index  - move_left) >= left_x ) and ((column_index  - move_left) <= medial_region_end):
                                distace_Left.append(down_point -upper_point)

                            elif  ((column_index - move_left) < lateral_region_start) and ((column_index - move_left) >  medial_region_end):
                                distace_Central.append(down_point -upper_point)
                        
                            else:
                                pass
                        except:
                            continue

# ___________________________________  End Vertical Distance For Left Region  ______________________________________ #
                
                if distace_Central == []:
                    pass
                
                else:
                    distace_Central.pop(0)

                self.average_medial_distance =  np.mean(distace_Left)
                self.average_central_distance =  np.mean (distace_Central)
                self.average_lateral_distance = np.mean(distace_Right)
                
                self.average_distance = (self.average_medial_distance + self.average_central_distance + self.average_lateral_distance) / 3
                self.average_distance_mm = self.average_distance / self.length_ratio


                # outer = 0
                # num_samples = int(self.JSW_width)
                
                # sample_x_Medial = []
                # sample_x_Central = []
                # sample_x_Lateral = []

                # for i in range(int(num_samples)):
                #     sample_x = int ( assumed_left + i)

                #     if assumed_left < sample_x < left_x:
                #         relevant_points = []
                #         vertical_distance = 0
                        
                #         outer += 1
                #         sample_x_Medial.append(vertical_distance)
                            
                #     elif left_x < sample_x < (medial_region_end):
                        
                #         relevant_points = [point for point in polygon[0] if point[0] == sample_x]
                        
                #         if relevant_points:
                #             max_y = max(point[1] for point in relevant_points)
                #             min_y = min(point[1] for point in relevant_points)
                            
                #             vertical_distance = max_y - min_y
                            
                #             if vertical_distance > 0 :
                                
                #                         sample_x_Medial.append(vertical_distance)
                                        
                            
                            
                            
                            
                            
                            
                #     if central_region_start < sample_x < central_region_end:
                        
                        
                #         relevant_points = [point for point in polygon[0] if point[0] == sample_x]
                        
                #         if relevant_points:
                #             max_y = max(point[1] for point in relevant_points)
                #             min_y = min(point[1] for point in relevant_points)
                            
                #             vertical_distance = max_y - min_y
                            
                #             if vertical_distance > 0 :
                #                 sample_x_Central.append(vertical_distance)
                        
                        

                                
                #     if right_x < sample_x < assumed_right:
                #         relevant_points = []
                #         vertical_distance = 0
                        
                #         outer += 1 
                        
                #         sample_x_Lateral.append(vertical_distance)

                #     elif  (lateral_region_start) < sample_x < right_x:
                        
                #         relevant_points = [point for point in polygon[0] if point[0] == sample_x]
                                                
                #         if relevant_points:
                #             max_y = max(point[1] for point in relevant_points)
                #             min_y = min(point[1] for point in relevant_points)
                            
                #             vertical_distance = max_y - min_y
                            
                #             if vertical_distance > 0 :
                #                     sample_x_Lateral.append(vertical_distance)
                            
                #     # print(f"At sample-x:{sample_x}, relevant_points: {relevant_points}")
                # # print(f"percentage: {self.percentage}")
                # # print(f"outer:{outer}")
                
                

                # sample_x_Medial.sort(reverse=True)
                # sample_x_Lateral.sort(reverse=True)




                # if len(sample_x_Medial) > len( sample_x_Lateral):
                #     x = int (0.9 * len(sample_x_Lateral))
                #     self.average_medial_distance  = np.mean(sample_x_Medial[int(0.5 * len(sample_x_Medial) - 0.5 * x) : int(0.5 * len(sample_x_Medial) + 0.5 * x)])
                #     self.average_lateral_distance  = np.mean(sample_x_Lateral[int((0.05)*len(sample_x_Lateral)) : int((0.95)*len(sample_x_Lateral))])
                
                # else:
                #     x = int (0.9 * len(sample_x_Medial))
                #     self.average_lateral_distance  = np.mean(sample_x_Lateral[int(0.5 * len(sample_x_Lateral) - 0.5 * x) : int(0.5 * len(sample_x_Lateral) + 0.5 * x)])
                #     self.average_medial_distance  = np.mean(sample_x_Medial[int((0.05)*len(sample_x_Medial)) : int((0.95)*len(sample_x_Medial))])
                
                # # self.average_medial_distance = (np.mean(sample_x_Medial))
                # self.average_central_distance = (np.mean(sample_x_Central))
                # # self.average_lateral_distance = (np.mean(sample_x_Lateral))
                
                
                
                self.medial_area = np.sum(distace_Left)
                self.central_area = np.sum(distace_Central)
                self.lateral_area = np.sum(distace_Right)

                self.JSN_Area_Total =  self.medial_area + self.central_area + self.lateral_area
                self.JSN_Area_Total_Squared_mm = self.JSN_Area_Total / self.area_ratio
                
                if (self.check_knee_literality):
                    pass
                else:
                    self.spare = self.average_medial_distance
                    self. average_medial_distance = self.average_lateral_distance
                    self.average_lateral_distance = self.spare

                print(f"\033[92m               ___________________________________________________________       \033[0m")

                print(f"\033[92mAvg V.Distance (Medial): {self.average_medial_distance} Pixel.\033[0m")
                print(f"\033[92mAvg V.Distance (Central): {self.average_central_distance} Pixel.\033[0m")
                print(f"\033[92mAvg V.Distance (Lateral): {self.average_lateral_distance} Pixel.\033[0m")
                
                print(f"\033[92m               ___________________________________________________________       \033[0m")

                self.average_medial_distance_mm = self.average_medial_distance / self.length_ratio
                self.average_central_distance_mm = self.average_central_distance / self.length_ratio
                self.average_lateral_distance_mm = self.average_lateral_distance / self.length_ratio
                
                print(f"\033[92mAvg V.Distance (Medial): {self.average_medial_distance_mm} mm.\033[0m")
                print(f"\033[92mAvg V.Distance (Central): {self.average_central_distance_mm} mm.\033[0m")
                print(f"\033[92mAvg V.Distance (Lateral): {self.average_lateral_distance_mm} mm.\033[0m")
                
                
                print(f"\033[92m               ___________________________________________________________       \033[0m")

                
                if (self.Tibial_width != 0):
                    
                    self.Medial_ratio = self.average_medial_distance/self.Tibial_width
                    self.Central_ratio = self.average_central_distance/self.Tibial_width
                    self.Lateral_ratio = self.average_lateral_distance/self.Tibial_width
                    
                    print(f"\033[92mAvg Tibial V.Distance ratio (Medial): {self.Medial_ratio}.\033[0m")
                    print(f"\033[92mAvg Tibial V.Distance ratio (Central): {self.Central_ratio}.\033[0m")
                    print(f"\033[92mAvg Tibial V.Distance ratio (Lateral): {self.Lateral_ratio}.\033[0m")
                    print(f"\033[92m               ___________________________________________________________       \033[0m")
                    
                else:
                    pass
                    print("\033[91mTibial Width: {self.Tibial_width} Pixel.\033[0m")
                
                painter.end()
                target_label.setPixmap(pixmap)
                target_label.setScaledContents(True)
            else:
                print("\033[91mNo points in the polygon to calculate distances from.\033[0m")
        else:
            pass




    def draw_subregion(self, painter, pixmap, polygon, start_x, end_x, color):
        subregion_path = QPainterPath()
        subregion_path.addRect(start_x, 0, end_x - start_x, pixmap.height())
        painter.setClipPath(subregion_path)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))
  
    def calculate_three_subregions_areas(self):
        return self.medial_area, self.central_area, self.lateral_area
  
    def fill_colored_polygon(self , image, two_dim_array, color=(0, 0, 0)):
            two_dim_array = np.array([two_dim_array], dtype=np.int32)
            cv2.fillPoly(image, [two_dim_array], color=color)


    def YOLO_CReg_predict(self, img):
            if self.splash_screen.model_YOLO_CenteralReg is not None and img is not None:
                predict_imageCReg = self.splash_screen.model_YOLO_CenteralReg.predict(img)
                yolo_results_CReg = self.get_coordinate_Predict_image(predict_imageCReg)
                self.draw_yolo_roi_box_CReg(yolo_results_CReg, img)
                # return yolo_results_CReg
            else:
                return "\033[91mNo image or model available.\033[0m"
    
    def draw_yolo_roi_box_CReg(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                self.Centeral_Left_X, self.Centeral_Left_Y, self.Centeral_Right_X, self.Centeral_Right_Y, class_name, prob = result
                
                if prob > 0.1:
                    top_left = QPoint(self.Centeral_Left_X, self.Centeral_Left_Y)
                    bottom_right = QPoint(self.Centeral_Right_X, self.Centeral_Right_Y)
                    self.Centeral_Region_Width = self.Centeral_Right_X - self.Centeral_Left_X
                    self.Centeral_Region_Height = self.Centeral_Right_Y - self.Centeral_Left_Y
                    self.Xc = self.Centeral_Left_X + self.Centeral_Region_Width/2
                    self.Yc = self.Centeral_Left_Y + self.Centeral_Region_Height/2
                    self.Centeral_Region_Center = (self.Xc , self.Yc)
                    painter = QPainter(self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(2)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label.update()
                    print(f"\033[92mCenteral Region Width: = {self.Centeral_Region_Width} Pixel.\033[0m")
                    print(f"\033[92mCenteral Region Height: = {self.Centeral_Region_Height} Pixel.\033[0m")
                    print(f"\033[92mCenteral Region Center: = {self.Centeral_Region_Center}.\033[0m")
                    

    def YOLO_TibialW_predict(self, img):
                if self.splash_screen.model_YOLO_TibialWidth is not None and img is not None:
                    predict_imageTibialW = self.splash_screen.model_YOLO_TibialWidth.predict(img)
                    yolo_results_TibialW = self.get_coordinate_Predict_image(predict_imageTibialW)
                    self.draw_yolo_roi_box_TibialW(yolo_results_TibialW, img)
                    # return yolo_results_TibialW
                else:
                    return "\033[91mNo image or model available.\033[0m"
        
    def draw_yolo_roi_box_TibialW(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                self.start_Tibial_width, y1, self.end_Tibial_width, y2, class_name, prob = result
                if prob > 0.1:
                    top_left = QPoint(self.start_Tibial_width, y1)
                    bottom_right = QPoint(self.end_Tibial_width, y2)
                    self.Tibial_width = self.end_Tibial_width-self.start_Tibial_width
                    self.Tibial_width_Predicted_Area = self.Tibial_width * (y2-y1)
                    self.Tibial_width_Predicted_Area_mm = self.Tibial_width_Predicted_Area / self.area_ratio
                    painter = QPainter(self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(2)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label.update()
                    print(f"\033[92mTibial Width: = {self.Tibial_width} Pixel.\033[0m")
                    
                    
    def YOLO_predict(self, img):
        if self.splash_screen.model is not None and img is not None:
            predict_image = self.splash_screen.model.predict(img)
            yolo_results = self.get_coordinate_Predict_image(predict_image)
            self.draw_yolo_roi_box(yolo_results, img)
            return yolo_results
        else:
            return "\033[91mNo image or model available.\033[0m"
        
        

    def get_coordinate_Predict_image(self, predict_image):
        rlts = predict_image [0]
        output = [] 

        for box  in rlts.boxes:
            x1 ,y1,x2,y2 = [
                round(x) for x in box.xyxy[0].tolist()
            ]
            class_id = box.cls[0].item()
            prob =round(box.conf[0].item(),2)
            output.append([
                x1 ,y1,x2,y2 ,rlts.names[class_id] , prob
            ])
        
        return  output
    
    def draw_yolo_roi_box(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                x1, y1, x2, y2, class_name, prob = result
                if prob > 0.1:
                    top_left = QPoint(x1, y1)
                    bottom_right = QPoint(x2, y2)
                    painter = QPainter(self.YOLO_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(4)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.YOLO_label.update()



    def save_PNGs(self):

        labels = [
            (self.image_label, "Manual Feature Extractor image.png"),
            (self.JSN_label, "JSN_label_Sub-Regions.png"),
            (self.YOLO_label, "YOLO_ROI.png"),
            (self.Intensity_label, "Intensity_label.png"),
            (self.Binarization_label, "Binarization_label.png"),
            (self.edge_label, "canny-edge-label.png"),
            (self.PredictedJSNOriginalWindow.Predicted_JSN_label, "JSN-Region.png"),
            (self.HistogramOfOrientedGradientsImage.HOG_image_label, "HOG Image.png")
        ]

        for label, filename in labels:
            pixmap = label.grab()
            img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            resized_image = cv2.resize(image, (224, 224))
            cv2.imwrite(f"Outputs/{filename}", resized_image)

                    #  _____________________________ setter ____________________________________  #

    
    def set_image(self, image):
        self.Conventional_image = image
    
    def set_path(self, path):
        self.Conventional_image_path = path
    
    def set_average_medial_distance(self, average_medial_distance):
        self.average_medial_distance = average_medial_distance
    
    def set_average_central_distance(self, average_central_distance):
        self.average_central_distance = average_central_distance
    
    def set_average_lateral_distance(self, average_lateral_distance):
        self.average_lateral_distance = average_lateral_distance
                                                                   
    def set_average_medial_distance_mm(self, average_medial_distance_mm):
        self.average_medial_distance_mm = average_medial_distance_mm
    
    def set_average_central_distance_mm(self, average_central_distance_mm):
        self.average_central_distance_mm = average_central_distance_mm
    
    def set_average_lateral_distance_mm(self, average_lateral_distance_mm):
        self.average_lateral_distance_mm = average_lateral_distance_mm
    
    def set_Medial_ratio(self, Medial_ratio):
        self.Medial_ratio = Medial_ratio
    
    def set_Central_ratio(self, Central_ratio):
        self.Central_ratio = Central_ratio
    
    def set_Lateral_ratio(self, Lateral_ratio):
        self.Lateral_ratio = Lateral_ratio
    
    def set_average_distance(self, average_distance):
        self.average_distance = average_distance
    
    def set_average_distance_mm(self, average_distance_mm):
        self.average_distance_mm = average_distance_mm
    
    def set_madeial_area(self, madeial_area):
        self.medial_area =madeial_area
    
    def set_central_area(self, central_area):
        self.central_area = central_area
    
    def set_lateral_area(self, lateral_area):
        self.lateral_area = lateral_area
    
    def set_medial_area_Squaredmm(self, medial_area_Squaredmm):
        self.medial_area_Squaredmm = medial_area_Squaredmm
    
    def set_central_area_Squaredmm(self, central_area_Squaredmm):
        self.central_area_Squaredmm = central_area_Squaredmm
    
    def set_lateral_area_Squaredmm(self, lateral_area_Squaredmm):
        self.lateral_area_Squaredmm = lateral_area_Squaredmm
    
    def set_intensity_mean(self, intensity_mean):
        self.intensity_mean = intensity_mean
        
    def set_intensity_stddev(self, intensity_stddev):
        self.intensity_stddev = intensity_stddev
    
    def set_intensity_skewness(self, intensity_skewness):
        self.intensity_skewness = intensity_skewness
    
    def set_intensity_kurtosis(self, intensity_kurtosis):
        self.intensity_kurtosis = intensity_kurtosis
    
    def set_cooccurrence_properties(self, cooccurrence_properties):
        self.cooccurrence_properties = cooccurrence_properties
        
    def set_lbp_features(self, lbp_features):
        self.lbp_features = lbp_features
    
    def set_lbp_variance(self, lbp_variance):
        self.lbp_variance = lbp_variance
    
    def set_lbp_entropy(self, lbp_entropy):
        self.lbp_entropy = lbp_entropy
    
    
    def set_ltp_features(self, ltp_features):
        self.ltp_features = ltp_features
        
    def set_ltp_variance(self, ltp_variance):
        self.ltp_variance = ltp_variance
    
    def set_ltp_entropy(self, ltp_entropy):
        self.ltp_entropy = ltp_entropy
        
                    #  _____________________________ getter ____________________________________  #
        
    def get_image(self):
        return self.Conventional_image
    
    def get_path(self):
        return self.Conventional_image_path
              
    def get_average_medial_distance(self):
        return self.average_medial_distance
    
    def get_average_central_distance(self):
        return self.average_central_distance
    
    def get_average_lateral_distance(self):
        return self.average_lateral_distance
                                                                   
    def get_average_medial_distance_mm(self):
        return self.average_medial_distance_mm
    
    def get_average_central_distance_mm(self):
        return self.average_central_distance_mm
    
    def get_average_lateral_distance_mm(self):
        return self.average_lateral_distance_mm
    
    def get_Medial_ratio(self):
        return self.Medial_ratio
    
    def get_Central_ratio(self):
        return self.Central_ratio
    
    def get_Lateral_ratio(self):
        return self.Lateral_ratio
    
    def get_average_distance(self):
        return self.average_distance
    
    def get_average_distance_mm(self):
        return self.average_distance_mm
    
    def get_madeial_area(self):
        return self.medial_area
    
    def get_central_area(self):
        return self.central_area
    
    def get_lateral_area(self):
        return self.lateral_area
    
    def get_medial_area_Squaredmm(self):
        return self.medial_area_Squaredmm
    
    def get_central_area_Squaredmm(self):
        return self.central_area_Squaredmm
    
    def get_lateral_area_Squaredmm(self):
        return self.lateral_area_Squaredmm
    
    def get_intensity_mean(self):
        return self.intensity_mean
        
    def get_intensity_stddev(self):
        return self.intensity_stddev
    
    def get_intensity_skewness(self):
        return self.intensity_skewness
    
    def get_intensity_kurtosis(self):
        return self.intensity_kurtosis
    
    def get_cooccurrence_properties(self):
        return self.cooccurrence_properties
        
    def get_lbp_features(self):
        return self.lbp_features
    
    def get_lbp_variance(self):
        return self.lbp_variance
    
    def get_lbp_entropy(self):
        return self.lbp_entropy
    
    def get_ltp_features(self):
        return self.ltp_features
        
    def get_ltp_variance(self):
        return self.ltp_variance
    
    def get_ltp_entropy(self):
        return self.ltp_entropy
    
                    #  ____________________________________________________________________________  #
    

    def calculate_histogram(self):
        if self.user_cropped and self.cropped_images:
            cropped_image = self.cropped_images[-1]
            if cropped_image is not None and not np.isnan(cropped_image).any() and cropped_image.size > 0:
                histogram = cv2.calcHist([cropped_image], [0], None, [256], [0, 256])
                self.histogram_window.set_histogram(histogram)
                self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(cropped_image)
                top_left, bottom_right = self.calculate_cropped_rect_coords(cropped_image)
                self.display_intensity_stats_coordinates(self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis, top_left, bottom_right)
        elif not self.user_cropped:
            histogram = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            self.histogram_window.set_histogram(histogram)


    def calculate_cropped_rect_coords(self, cropped_image):
        if cropped_image is not None:
            original_image_size = self.image.shape[:2]
            crop_rect = self.image_label.crop_rect
            top_left = (
                int(crop_rect.left() * original_image_size[1] / self.image_label.width()),
                int(crop_rect.top() * original_image_size[0] / self.image_label.height())
            )
            bottom_right = (
                int(crop_rect.right() * original_image_size[1] / self.image_label.width()),
                int(crop_rect.bottom() * original_image_size[0] / self.image_label.height())
            )
            return top_left, bottom_right
        else:
            return None, None

    def perform_edge_detection(self, img):
        if img is not None:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.perform_intensity_normalization:
                equalized_image = cv2.equalizeHist(gray_image)
                edges = cv2.Canny(equalized_image, 50, 150)
            else:
                edges = cv2.Canny(gray_image, 50, 150)

            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_edge_image(edges_rgb)

    def display_edge_image(self, img):
        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.edge_label.setPixmap(QPixmap.fromImage(q_img))
            self.edge_label.setScaledContents(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.image is not None:
                self.image_label.crop_rect.setTopLeft(event.pos())
                self.image_label.crop_rect.setBottomRight(event.pos())
                self.image_label.mouse_pressed = True

    def mouseMoveEvent(self, event):
        if self.image_label and hasattr(self.image_label, 'mouse_pressed') and self.image_label.mouse_pressed:
            self.image_label.crop_rect.setBottomRight(event.pos())
            self.image_label.update()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.image is not None:
                self.image_label.mouse_pressed = False
                self.image_label.crop_rect.setBottomRight(event.pos())
                
                original_image_size = self.image.shape[:2]
                
                top_left = (int(self.image_label.crop_rect.left() * original_image_size[1] / self.image_label.width()),
                            int(self.image_label.crop_rect.top() * original_image_size[0] / self.image_label.height()))
                bottom_right = (int(self.image_label.crop_rect.right() * original_image_size[1] / self.image_label.width()),
                                int(self.image_label.crop_rect.bottom() * original_image_size[0] / self.image_label.height()))
                
                
                cropped_image = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                if cropped_image is not None and not np.isnan(cropped_image).any() and cropped_image.size > 0:
                    self.cropped_images.append(cropped_image)
                    self.calculate_histogram()
                    self.user_cropped = True
                    self.calculate_and_save_all_parameters(cropped_image, self.image_path,top_left,bottom_right)
                            
    # def dragEnterEvent(self, event):
    #     mime_data = event.mimeData()
    #     if mime_data.hasUrls() and mime_data.urls()[0].isLocalFile():
    #         event.acceptProposedAction()


    # def dropEvent(self, event):
    #     file_path = event.mimeData().urls()[0].toLocalFile()
    #     self.load_image(file_path)

    def mouseDoubleClickEvent(self, event):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("PNG Images (*.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                image_path = file_paths[0]
                self.load_image(image_path)
                    
    def equalize_histogram(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
            self.display_equalized_image(equalized_image_rgb)

    def display_equalized_image(self, img):
        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.Intensity_label.setPixmap(QPixmap.fromImage(q_img))
            self.Intensity_label.setScaledContents(True)

    def binarization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            ret, binary_otsu = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_binarized_image(binary_otsu)
            return binary_otsu
        
    def display_binarized_image(self, img):
        if img is not None:
            if len(img.shape) == 2:
                height, width = img.shape
                bytes_per_line = width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            self.Binarization_label.setPixmap(pixmap)
            self.Binarization_label.setScaledContents(True)
            
            
    def on_button_click(self):
            self.Conventional_image = None
            self.Conventional_image_path = None
            self.image = None
            self.image_path = None
            self.cropped_images = []
            self.user_cropped = False
            self.setAcceptDrops(True)
            self.image_label.clear()
            self.JSN_label.clear()
            self.YOLO_label.clear()
            self.Intensity_label.clear()
            self.Binarization_label.clear()
            self.edge_label.clear()
            self.histogram_window.Histogram_Intensity_label.clear()
            self.LBPandLTPImgsWindow.LBPImgWindow_label.clear()
            self.LBPandLTPImgsWindow.LTPImgWindow_label.clear()
            self.LBPHistogramWindow.LBPHistogramWindow_label.clear()
            self.LTPHistogramWindow.LTPHistogramWindow_label.clear()
            self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label.clear()
            self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label.clear()
            self.PredictedJSNOriginalWindow.Predicted_JSN_label.clear()
            self.HistogramOfOrientedGradientsImage.HOG_image_label.clear()
            self.HOGHistogram.HOGHistogram_label.clear()
            self.image_label.setText("Manual Feature Extractor")
            self.JSN_label.setStyleSheet("background-image: url(imgs/Feature_Extraction.png);background-image: no-repeat;")
            self.YOLO_label.setText("ROI Detector")
            self.Intensity_label.setText("Intensity normalization")
            self.Binarization_label.setText("Binarization")
            self.edge_label.setText("Canny Edge Detection")
            self.histogram_window.Histogram_Intensity_label.setText("Intensity Histogram Window")
            self.LBPandLTPImgsWindow.LBPImgWindow_label.setText("LBP")
            self.LBPandLTPImgsWindow.LTPImgWindow_label.setText("LTP")
            self.LBPHistogramWindow.LBPHistogramWindow_label.setText("LBP Histogram Window")
            self.LTPHistogramWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
            self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label.setText("Centeral Region Width Image")
            self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label.setText("Tibial Width Image")
            self.PredictedJSNOriginalWindow.Predicted_JSN_label.setText("Predicted JSN Original Image")
            self.HistogramOfOrientedGradientsImage.HOG_image_label.setText("HOG Image")
            self.HOGHistogram.HOGHistogram_label.setText("HOG Histogram")
            
    def toggle_intensity_normalization(self, state):
        if state == Qt.Checked:
            self.perform_intensity_normalization = True

            
        else:

            self.perform_intensity_normalization = False
            self.Intensity_label.clear()
            self.Binarization_label.clear()
            self.LBPandLTPImgsWindow.LBPImgWindow_label.clear()
            self.LBPandLTPImgsWindow.LTPImgWindow_label.clear()
            self.LBPHistogramWindow.LBPHistogramWindow_label.clear()
            self.LTPHistogramWindow.LTPHistogramWindow_label.clear()
            self.HistogramOfOrientedGradientsImage.HOG_image_label.clear()
            self.HOGHistogram.HOGHistogram_label.clear()
            self.Binarization_label.setText("Binarization")
            self.Intensity_label.setText("Intensity normalization")
            self.LBPandLTPImgsWindow.LBPImgWindow_label.setText("LBP")
            self.LBPandLTPImgsWindow.LTPImgWindow_label.setText("LTP")
            self.LBPHistogramWindow.LBPHistogramWindow_label.setText("LBP Histogram Window")
            self.LTPHistogramWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
            self.HistogramOfOrientedGradientsImage.HOG_image_label.setText("HOG Image")
            self.HOGHistogram.HOGHistogram_label.setText("HOG Histogram")
        if self.image_path is not None:
            self.load_image(self.image_path)
        else:
            pass

    def update_histogram_display(self):
        self.calculate_histogram()


    def calculate_intensity_stats(self, image):
        if image is not None:
            self.intensity_mean = np.mean(image)
            self.intensity_stddev = np.std(image)
            self.intensity_skewness = skew(image, axis=None)
            self.intensity_kurtosis = kurtosis(image, axis=None)
            
            return self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis
        else:
            return None, None, None, None


    def display_intensity_stats_coordinates(self, mean, stddev, intensity_skewness, intensity_kurtosis, top_left, bottom_right):
        if (mean is not None and stddev is not None) and (intensity_skewness is not None and intensity_kurtosis is not None):
            stats_text = f"\n\nIntensity Mean: {mean:.2f}\nIntensity StdDev: {stddev:.2f}\n" + \
             f"Intensity Skewness: {intensity_skewness:.2f}\nIntensity Kurtosis: {intensity_kurtosis:.2f}\n\n" + \
             f"Left-Top: {top_left[0]:.2f}, {top_left[1]:.2f}\n" + \
             f"Right-Bottom: {bottom_right[0]:.2f}, {bottom_right[1]:.2f}"

            self.Intensity_label.setText(stats_text)
        else:
            self.Intensity_label.clear()


    def calculate_cooccurrence_parameters(self, image):
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        max_probability = np.max(glcm)

        properties = {
            "contrast": graycoprops(glcm, 'contrast').mean(),
            "energy": graycoprops(glcm, 'energy').mean(),
            "correlation": graycoprops(glcm, 'correlation').mean(),
            "homogeneity": graycoprops(glcm, 'homogeneity').mean(),
            "dissimilarity": graycoprops(glcm, 'dissimilarity').mean(),
            "ASM": graycoprops(glcm, 'ASM').mean(),
            "max_probability": max_probability,
        }

        return properties

    
    def calculate_cooccurrence_parameters_YOLO(self, image):
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            max_probability = np.max(glcm)

            properties = {
                "contrast": graycoprops(glcm, 'contrast').mean(),
                "energy": graycoprops(glcm, 'energy').mean(),
                "correlation": graycoprops(glcm, 'correlation').mean(),
                "homogeneity": graycoprops(glcm, 'homogeneity').mean(),
                "dissimilarity": graycoprops(glcm, 'dissimilarity').mean(),
                "ASM": graycoprops(glcm, 'ASM').mean(),
                "max_probability": max_probability,
            }

            return properties
    

    def calculate_lbp_features(self, image):
        radius = 1
        n_points = 8 * radius
        lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        # lbp_histogram = lbp_histogram / (lbp_histogram.sum() + 1e-5)
        self.LBPHistogramWindow.set_lbp_histogram(lbp_histogram)
        return lbp_histogram, lbp_image


    def calculate_lbp_features_YOLO(self, image):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            radius = 1
            n_points = 8 * radius
            lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
            lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            # lbp_histogram = lbp_histogram / (lbp_histogram.sum() + 1e-5)
            self.LBPHistogramWindow.set_lbp_histogram(lbp_histogram)
            return lbp_histogram, lbp_image

    def calculate_ltp_features(self, image, num_bins = 10):
        h, w = image.shape
        ltp_image = np.zeros((h, w), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center_pixel = image[y, x]
                binary_pattern = 0
                binary_pattern |= (image[y - 1, x - 1] >= center_pixel) << 7
                binary_pattern |= (image[y - 1, x] >= center_pixel) << 6
                binary_pattern |= (image[y - 1, x + 1] >= center_pixel) << 5
                binary_pattern |= (image[y, x + 1] >= center_pixel) << 4
                binary_pattern |= (image[y + 1, x + 1] >= center_pixel) << 3
                binary_pattern |= (image[y + 1, x] >= center_pixel) << 2
                binary_pattern |= (image[y + 1, x - 1] >= center_pixel) << 1
                binary_pattern |= (image[y, x - 1] >= center_pixel)

                ltp_image[y, x] = binary_pattern
        
        num_bins += 1
        ltp_histogram, _ = np.histogram(ltp_image, bins=np.arange(0, num_bins), range=(0, num_bins))
        # ltp_histogram = ltp_histogram / (ltp_histogram.sum() + 1e-5)
        self.LTPHistogramWindow.set_LTPHistogramWindow_histogram(ltp_histogram)
        return ltp_histogram, ltp_image


    def calculate_ltp_features_YOLO(self, image, num_bins=10):
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        ltp_image = np.zeros((h, w), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center_pixel = image[y, x]
                binary_pattern = 0
                binary_pattern |= (image[y - 1, x - 1] >= center_pixel) << 7
                binary_pattern |= (image[y - 1, x] >= center_pixel) << 6
                binary_pattern |= (image[y - 1, x + 1] >= center_pixel) << 5
                binary_pattern |= (image[y, x + 1] >= center_pixel) << 4
                binary_pattern |= (image[y + 1, x + 1] >= center_pixel) << 3
                binary_pattern |= (image[y + 1, x] >= center_pixel) << 2
                binary_pattern |= (image[y + 1, x - 1] >= center_pixel) << 1
                binary_pattern |= (image[y, x - 1] >= center_pixel)

                ltp_image[y, x] = binary_pattern

        num_bins += 1
        ltp_histogram, _ = np.histogram(ltp_image, bins=np.arange(0, num_bins), range=(0, num_bins))
        # ltp_histogram = ltp_histogram / (ltp_histogram.sum() + 1e-5)
        self.LTPHistogramWindow.set_LTPHistogramWindow_histogram(ltp_histogram)
        return ltp_histogram, ltp_image




    def compute_hog_features(self, image, cell_size=(8, 8), block_size=(1, 1), bins=9):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = image / 255.0

            gradients_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradients_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

            gradients_magnitude = np.sqrt(gradients_x**2 + gradients_y**2)
            gradients_orientation = np.arctan2(gradients_y, gradients_x) * (180 / np.pi)

            hog_bins = np.zeros(bins)
            hog_image = np.zeros_like(image, dtype=float)
            
            

            for i in range(0, image.shape[0], cell_size[0]):
                for j in range(0, image.shape[1], cell_size[1]):
                    cell_magnitude = gradients_magnitude[i:i + cell_size[0], j:j + cell_size[1]]
                    cell_orientation = gradients_orientation[i:i + cell_size[0], j:j + cell_size[1]]

                    histogram = np.zeros(bins)
                    for k in range(bins):
                        angles = (cell_orientation >= k * (180 / bins)) & (cell_orientation < (k + 1) * (180 / bins))
                        histogram[k] = np.sum(cell_magnitude[angles])

                    hog_bins += histogram
                    hog_image[i:i + cell_size[0], j:j + cell_size[1]] = histogram.argmax() * (180 / bins)

            epsilon = 1e-6
            hog_bins /= (np.linalg.norm(hog_bins) + epsilon)
            
            
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            print(f"\033[92mlength of Histogram Of Oriented Gradients:{len(hog_bins)} bins\n\033[0m")
            print(f"\033[92mHOG Feature vector:{(hog_bins)}\n\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")


            self.HOGHistogram.set_HOG_Histogram(hog_bins)
            self.HistogramOfOrientedGradientsImage.set_HOG_Image(hog_image)
            
            return hog_bins

        else:
            return []






    # def compute_hog_features(self, image, cell_size= (8,8), block_size = (1,1)):
    #     if image is not None:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    #         image = image / 255.0
            
    #         h, w = image.shape

    #         gradients_array = np.zeros((w, h))
    #         orientation_array = np.zeros((w, h))
        
            
    #         gradients_SUM_0 = 0
    #         gradients_SUM_20 = 0
    #         gradients_SUM_40 = 0
    #         gradients_SUM_60 = 0
    #         gradients_SUM_80 = 0
    #         gradients_SUM_100 = 0
    #         gradients_SUM_120 = 0
    #         gradients_SUM_140 = 0
    #         gradients_SUM_160 = 0
    #         gradients_SUM_180 = 0
        
    #         for y in range(1, h - 1):
    #             for x in range(1, w - 1):
                    
    #                 # center_pixel = image[y, x]
                    
    #                 gradients_x = np.abs(image[y, x+1] - image[y, x-1])
    #                 gradients_y = np.abs(image[y+1, x] - image[y-1, x])
                    
    #                 gradients_x = gradients_x if gradients_x != 0 else 1e-6
                
    #                 gradients = np.sqrt(gradients_x**2 + gradients_y**2)
    #                 orientation = np.arctan(gradients_y / gradients_x) * (180 / np.pi)
                    
    #                 gradients_array[x, y] = gradients
    #                 orientation_array[x, y] = orientation
                    
                    
                    
    #                 if orientation_array[x, y] == 0:
    #                     gradients_SUM_0 += gradients_array[x, y]
                    
    #                 if orientation_array[x, y] == 20:
    #                     gradients_SUM_20 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 40:
    #                     gradients_SUM_40 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 60:
    #                     gradients_SUM_60 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 80:
    #                     gradients_SUM_80 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 100:
    #                     gradients_SUM_100 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 120:
    #                     gradients_SUM_120 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 140:
    #                     gradients_SUM_140 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 160:
    #                     gradients_SUM_160 += gradients_array[x, y]
                        
    #                 if orientation_array[x, y] == 180:
    #                     gradients_SUM_180 += gradients_array[x, y]
                    
                    
    #                 if 0 < orientation_array[x, y] < 20:
    #                     gradients_SUM_0 += ((orientation_array[x, y] - 0) / 20) * gradients_array[x, y]
    #                     gradients_SUM_20 += ((20 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 20 < orientation_array[x, y] < 40:
    #                     gradients_SUM_20 += ((orientation_array[x, y] - 20) / 20) * gradients_array[x, y]
    #                     gradients_SUM_40 += ((40 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 40 < orientation_array[x, y] < 60:
    #                     gradients_SUM_40 += ((orientation_array[x, y] - 40) / 20) * gradients_array[x, y]
    #                     gradients_SUM_60 += ((60 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 60 < orientation_array[x, y] < 80:
    #                     gradients_SUM_60 += ((orientation_array[x, y] - 60) / 20) * gradients_array[x, y]
    #                     gradients_SUM_80 += ((80 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 80 < orientation_array[x, y] < 100:
    #                     gradients_SUM_80 += ((orientation_array[x, y] - 80) / 20) * gradients_array[x, y]
    #                     gradients_SUM_100 += ((100 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
                        
    #                 if 100 < orientation_array[x, y] < 120:
    #                     gradients_SUM_100 += ((orientation_array[x, y] - 100) / 20) * gradients_array[x, y]
    #                     gradients_SUM_120 += ((120 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 120 < orientation_array[x, y] < 140:
    #                     gradients_SUM_120 += ((orientation_array[x, y] - 120) / 20) * gradients_array[x, y]
    #                     gradients_SUM_140 += ((140 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 140 < orientation_array[x, y] < 160:
    #                     gradients_SUM_140 += ((orientation_array[x, y] - 140) / 20) * gradients_array[x, y]
    #                     gradients_SUM_160 += ((160 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                        
    #                 if 160 < orientation_array[x, y] < 180:
    #                     gradients_SUM_160 += ((orientation_array[x, y] - 160) / 20) * gradients_array[x, y]
    #                     gradients_SUM_180 += ((180 - orientation_array[x, y]) / 20) * gradients_array[x, y]
                    
                    
            
    #         print(f"gradients_array :{gradients_array}")
    #         print(f"orientation_array :{orientation_array}")
            
            
    #         print(f"gradients_array range: ({np.min(gradients_array)}, {np.max(gradients_array)})")
    #         print(f"orientation_array range: ({np.min(orientation_array)}, {np.max(orientation_array)})")

    #         self.hog_bins = [
    #                 gradients_SUM_0,
    #                 gradients_SUM_20,
    #                 gradients_SUM_40,
    #                 gradients_SUM_60,
    #                 gradients_SUM_80,
    #                 gradients_SUM_100,
    #                 gradients_SUM_120,
    #                 gradients_SUM_140,
    #                 gradients_SUM_160,
    #                 gradients_SUM_180
    #                 ]
            
               
    #         k = np.sqrt(np.sum(np.array(self.hog_bins) ** 2))
            
    #         print(f"beforn norm: {self.hog_bins}")
            
    #         self.hog_bins /= k   
            
    #         print(f"after norm: {self.hog_bins}")
            
            
            
    #         return self.hog_bins

    #     else:
    #         return []
            
            
    def calculate_and_save_all_parameters(self, cropped_image, image_path,top_left,bottom_right):
            if cropped_image is not None:
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                self.cooccurrence_properties = self.calculate_cooccurrence_parameters(cropped_image_gray)
                self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(cropped_image_gray)
                self.lbp_features, self.lbp_image = self.calculate_lbp_features(cropped_image_gray)
                self.ltp_features, self.ltp_image = self.calculate_ltp_features(cropped_image_gray)               
                self.LBPandLTPImgsWindow.set_lbp_image(self.lbp_image)
                self.LBPandLTPImgsWindow.set_ltp_image(self.ltp_image)
                self.lbp_variance = np.var(self.lbp_features)
                self.ltp_variance = np.var(self.ltp_features)
                self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                csv_file_path = "Outputs/statistics_Manual.csv"
                write_header = not os.path.exists(csv_file_path)
                try:
                    with open(csv_file_path, mode="a", newline="") as csv_file:
                        fieldnames = ["Image Name", "Class","Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                    [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"]+ [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"]
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        row_data = {
                            "Image Name": file_name,
                            "Class":self.folder_name,
                            "Crop Top-Left (X, Y)": f"({top_left[0]}, {top_left[1]})",
                            "Crop Bottom-Right (X, Y)": f"({bottom_right[0]}, {bottom_right[1]})",
                            "Mean": self.intensity_mean,
                            "Sigma": self.intensity_stddev,
                            "Skewness": self.intensity_skewness,
                            "Kurtosis": self.intensity_kurtosis,
                            **self.cooccurrence_properties,
                            **{f"LBP_{i}": self.lbp_features[i] for i in range(len(self.lbp_features))},
                            "LBP_Variance": self.lbp_variance,
                            "LBP_Entropy": self.lbp_entropy,
                            **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                            "LTP_Variance": self.ltp_variance,
                            "LTP_Entropy": self.ltp_entropy
                            
                        }
                        writer.writerow(row_data)
                        
                except Exception as e:
                        print("\033[91mError writing to statistics_Manual.csv:\033[0m", str(e))
            else:
                self.Intensity_label.clear()

    def calculate_and_save_features_YOLO(self, yolo_results, image_path, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                x1, y1, x2, y2, class_name, self.prob = result
                if self.prob > 0.1:
                    roi_cropped_image = img[y1:y2, x1:x2]
                    if roi_cropped_image is not None:
                        self.cooccurrence_properties = self.calculate_cooccurrence_parameters_YOLO(roi_cropped_image)
                        self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(roi_cropped_image)
                        self.hog_bins = self.compute_hog_features(roi_cropped_image)
                        # self.hog_bins = self.compute_hog_features(img)
                        self.lbp_features, _ = self.calculate_lbp_features_YOLO(roi_cropped_image)
                        self.ltp_features, _ = self.calculate_ltp_features_YOLO(roi_cropped_image)
                        self.lbp_variance = np.var(self.lbp_features)
                        self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                        if self.ltp_features is not None:
                            self.ltp_variance = np.var(self.ltp_features)
                            self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                        else:
                            self.ltp_variance = None
                            self.ltp_entropy = None
                        file_name = os.path.splitext(os.path.basename(image_path))[0]
                        csv_file_path_yolo = "Outputs/statistics_YOLO.csv"
                        write_header = not os.path.exists(csv_file_path_yolo)
                        try:
                            with open(csv_file_path_yolo, mode="a", newline="") as csv_file:
                                fieldnames = ["Image Name", "Class", "percentage", "Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "YOLO_Probability", "Tibial Width (Pixel)", "Tibial Width (mm)", "JSN Avg V.Distance (Pixel)", "JSN Avg V.Distance (mm)", "Medial_distance (Pixel)", "Central_distance (Pixel)", "Lateral_distance (Pixel)","Medial_distance (mm)", "Central_distance (mm)", "Lateral_distance (mm)", "Tibial_Medial_ratio", "Tibial_Central_ratio", "Tibial_Lateral_ratio", "JSN Area (Squared Pixel)", "JSN Area (Squared mm)", "Medial Area (Squared Pixel)", "Central Area (Squared Pixel)", "Lateral Area (Squared Pixel)","Medial Area (Squared mm)", "Central Area (Squared mm)", "Lateral Area (Squared mm)", "Medial Area (JSN Ratio)", "Central Area (JSN Ratio)", "Lateral Area (JSN Ratio)", "Medial Area Ratio TWPA (%)","Central Area Ratio TWPA (%)", "Lateral Area Ratio TWPA (%)","Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                            [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"] + \
                                            [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"] + \
                                            [f"HOG_{i}" for i in range(len(self.hog_bins))] 
                                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                                if write_header:
                                    writer.writeheader()
                                row_data = {
                                    "Image Name": file_name,
                                    "Class": self.folder_name,
                                    "percentage":self.percentage,
                                    "Crop Top-Left (X, Y)":f"({x1}, {y1})",
                                    "Crop Bottom-Right (X, Y)":f"({x2}, {y2})",
                                    "YOLO_Probability":f"{self.prob}",
                                    "Tibial Width (Pixel)": self.Tibial_width,
                                    "Tibial Width (mm)": self.Tibial_width/ self.length_ratio,
                                    "JSN Avg V.Distance (Pixel)": self.average_distance,
                                    "JSN Avg V.Distance (mm)": self.average_distance_mm,
                                    "Medial_distance (Pixel)":self.average_medial_distance,
                                    "Central_distance (Pixel)":self.average_central_distance,
                                    "Lateral_distance (Pixel)":self.average_lateral_distance,
                                    "Medial_distance (mm)":self.average_medial_distance_mm,
                                    "Central_distance (mm)":self.average_central_distance_mm,
                                    "Lateral_distance (mm)":self.average_lateral_distance_mm,
                                    "Tibial_Medial_ratio":self.Medial_ratio,
                                    "Tibial_Central_ratio":self.Central_ratio,
                                    "Tibial_Lateral_ratio":self.Lateral_ratio,
                                    "JSN Area (Squared Pixel)": self.JSN_Area_Total,
                                    "JSN Area (Squared mm)": self.JSN_Area_Total_Squared_mm,
                                    "Medial Area (Squared Pixel)":self.medial_area,
                                    "Central Area (Squared Pixel)":self.central_area,
                                    "Lateral Area (Squared Pixel)":self.lateral_area,                                                                                                
                                    "Medial Area (Squared mm)":self.medial_area_Squaredmm,
                                    "Central Area (Squared mm)":self.central_area_Squaredmm,
                                    "Lateral Area (Squared mm)":self.lateral_area_Squaredmm,
                                    "Medial Area (JSN Ratio)":self.medial_area_Ratio,
                                    "Central Area (JSN Ratio)":self.central_area_Ratio,
                                    "Lateral Area (JSN Ratio)":self.lateral_area_Ratio,
                                    "Medial Area Ratio TWPA (%)":self.medial_area_Ratio_TWPA,
                                    "Central Area Ratio TWPA (%)":self.central_area_Ratio_TWPA,
                                    "Lateral Area Ratio TWPA (%)":self.lateral_area_Ratio_TWPA,
                                    "Mean": self.intensity_mean,
                                    "Sigma": self.intensity_stddev,
                                    "Skewness": self.intensity_skewness,
                                    "Kurtosis": self.intensity_kurtosis,
                                    **self.cooccurrence_properties,
                                    **{f"LBP_{i}": self.lbp_features[i] for i in range(len(self.lbp_features))},
                                    "LBP_Variance": self.lbp_variance,
                                    "LBP_Entropy": self.lbp_entropy,
                                    **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                                    "LTP_Variance": self.ltp_variance,
                                    "LTP_Entropy": self.ltp_entropy,
                                    **{f"HOG_{i}": self.hog_bins[i] for i in range(len(self.hog_bins))}
                                    
                                }
                                writer.writerow(row_data)
                        except Exception as e:
                            print("\033[91mError writing to statistics_YOLO.csv:\033[0m", str(e))
                               
#  __________________________________________________ main() func. ____________________________________________________

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Feature_Extraction_and_Visualization_Screen()
    main_window.show()
    sys.exit(app.exec_())