'''
 * ________________________________________ Knee OsteoArthritis Diagnosis ________________________________________________
 *
 *  Features Extraction and Visuallization - Conventional CAD - Ai Automated CAD
 *  Created on: Friday Aug 15 2023
 *  Author    : Mohammad Sayed Zaky - BME Team 13
 '''
#  _____________________________________________ Libraries ____________________________________________________________
from PyQt5.QtWidgets import QApplication, QRadioButton, QScrollArea, QMainWindow, QToolTip,QGraphicsOpacityEffect, QFileDialog, QLabel,QMessageBox,QVBoxLayout,QPushButton, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon, QFont, QPainter, QPainterPath, QRegion, QPen, QPolygonF, QBrush, QPainter
from PyQt5.QtCore import Qt, pyqtSignal,QTimer, QRect, QDir,QRectF, QSize, QPoint, QPointF, QPropertyAnimation, QEasingCurve
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
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from ultralytics import YOLO
from Models.NewFeatureExtractionAndVisualizationModels.modelCNN import SimpleCNN
from joblib import load
from threading import Thread
from time import sleep
import pydicom
import tempfile
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import UID, generate_uid
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet201, DenseNet201_Weights
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#  _____________________________________________ Splash Screen ____________________________________________________________
class SplashScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Splash Screen")
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        Feature_Extraction_and_Visualization_Screen.image = None
        Feature_Extraction_and_Visualization_Screen.image_label = None
        
        self.background_label = QLabel(self)
        self.pixmap = QPixmap("imgs/blackSplashScreen.png")
        self.background_label.setPixmap(self.pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, 1920, 1080)
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.central_widget = QWidget(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.background_label)
#  ____________________________________ Including Models ________________________________________________
            # ___________ Feature Extraction and Visualization Models ____________#
        
        self.model =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_ROI detection.pt")
        self.model_instance_Segmentation =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_Instance Segmentation.pt")
        self.model_YOLO_CenteralReg =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_CenteralReg.pt")
        self.model_YOLO_TibialWidth =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_TibialWidth.pt")
        self.knee_Literality_model = SimpleCNN()
        self.knee_Literality_model.load_state_dict(torch.load('Models/NewFeatureExtractionAndVisualizationModels/Knee_Literality.pth'))
        self.knee_Literality_model.eval()
                        # ___________ Conventional CAD Models ____________#
                                    
        self.Random_Forest_Normale_Binary = load('Models/New Conventional Models/Random_Forest_Normale_Binary.pkl')
        self.scaler_Normalize_Binary = load('Models/New Conventional Models/scaler_Normalize_Binary.pkl')
        
        self.Random_Forest_Normalize_3_Class = load('Models/New Conventional Models/Random_Forest_Normalized_3_Class.pkl')
        self.scaler_Normalize_3_Class = load('Models/New Conventional Models/scaler_Normalize_3_Class.pkl')
        
        self.Random_Forest_Normalize_5_Class = load('Models/New Conventional Models/Random_Forest_Normalize_5_Class.pkl')
        self.scaler_Normalize_5_Class = load('Models/New Conventional Models/scaler_Normalize_5_Class.pkl')
        
        self.scaler_0_vs_1 = load("Models/New Conventional Models/scaler_0_vs_1.pkl")
        self.scaler_0_vs_2 = load("Models/New Conventional Models/scaler_0_vs_2.pkl")
        self.scaler_0_vs_3 = load("Models/New Conventional Models/scaler_0_vs_3.pkl")
        self.scaler_0_vs_4 = load("Models/New Conventional Models/scaler_0_vs_4.pkl")
        self.scaler_1_vs_2 = load("Models/New Conventional Models/scaler_1_vs_2.pkl")
        self.scaler_1_vs_3 = load("Models/New Conventional Models/scaler_1_vs_3.pkl")
        self.scaler_1_vs_4 = load("Models/New Conventional Models/scaler_1_vs_4.pkl")
        self.scaler_2_vs_3 = load("Models/New Conventional Models/scaler_2_vs_3.pkl")
        self.scaler_2_vs_4 = load("Models/New Conventional Models/scaler_2_vs_4.pkl")
        self.scaler_3_vs_4 = load("Models/New Conventional Models/scaler_3_vs_4.pkl")

        self.model_0_vs_1 = load("Models/New Conventional Models/model_0_vs_1.pkl")
        self.model_0_vs_2 = load("Models/New Conventional Models/model_0_vs_2.pkl")
        self.model_0_vs_3 = load("Models/New Conventional Models/model_0_vs_3.pkl")
        self.model_0_vs_4 = load("Models/New Conventional Models/model_0_vs_4.pkl")
        self.model_1_vs_2 = load("Models/New Conventional Models/model_1_vs_2.pkl")
        self.model_1_vs_3 = load("Models/New Conventional Models/model_1_vs_3.pkl")
        self.model_1_vs_4 = load("Models/New Conventional Models/model_1_vs_4.pkl")
        self.model_2_vs_3 = load("Models/New Conventional Models/model_2_vs_3.pkl")
        self.model_2_vs_4 = load("Models/New Conventional Models/model_2_vs_4.pkl")
        self.model_3_vs_4 = load("Models/New Conventional Models/model_3_vs_4.pkl")
        
                        # ___________ Ai Automated CAD Models ____________#

        self.pytorch_model_2C = self.load_pytorch_model('Models/New Ai Automated Models/pytorch_classification_model_2C.pth',2)
        self.pytorch_model_3C = self.load_pytorch_model('Models/New Ai Automated Models/pytorch_classification_model_3C.pth',3)
        self.pytorch_model_5C = self.load_pytorch_model('Models/New Ai Automated Models/pytorch_classification_model_5C.pth',5)
#  _______________________________________________________________________________________________________

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close_splash)
        self.timer.start(500)
        
    def close_splash(self):
        self.close()
    # _________________________________________resize to Full Screen______________________________________________
    def resizeEvent(self, event):
        pixmap = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(pixmap)
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())

    
    def load_pytorch_model(self, path,num_classes):
        model = densenet201(weights = None)
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier.in_features, num_classes),
            nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        )
        model.classifier = classifier

        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
#  ____________________________________________ CustomMessageBox _____________________________________________________
class CustomMessageBox(QMessageBox):
    def __init__(self):
        super(CustomMessageBox, self).__init__()
        self.setWindowIcon(QIcon("imgs/x.svg"))
        self.setText("Do you want to exit?")
        quit_button = self.addButton("Exit", QMessageBox.AcceptRole)
        self.addButton("Cancel", QMessageBox.RejectRole)
        quit_button.setStyleSheet("color: black;")
        self.setStyleSheet("QMessageBox { background-color: white; color: white; }")
class CustomMessageBox2(QMessageBox):
    def __init__(self):
        super(CustomMessageBox2, self).__init__()
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        self.setStyleSheet("QMessageBox { background-color: white; color: white; }")  
class CustomMessageBox3(QMessageBox):
    def __init__(self):
        super(CustomMessageBox3, self).__init__()
        self.setWindowIcon(QIcon("imgs/restart.svg"))
        self.setText("Do you want to restart?")
        quit_button = self.addButton("Restart", QMessageBox.AcceptRole)
        self.addButton("Cancel", QMessageBox.RejectRole)
        quit_button.setStyleSheet("color: black;")
        self.setStyleSheet("QMessageBox { background-color: white; color: white; }")
#  _____________________________________________ ImageLabel __________________________________________________________
class ImageLabel(QLabel):
    def __init__(self, parent, apply_effects=True):
        super().__init__(parent)
        self.crop_rect = QRect()
        self.mouse_pressed = False
        self.setScaledContents(True)
        
        self.apply_effects = apply_effects
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.OutQuart)
        self.leaveEvent(True)
        
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

    def enterEvent(self, event):
        if self.apply_effects:
            self.animation.stop()
            self.animation.setDirection(QPropertyAnimation.Forward)
            self.animation.start()

    def leaveEvent(self, event):
        if self.apply_effects:
            self.animation.stop()
            self.animation.setDirection(QPropertyAnimation.Backward)
            self.animation.start()
    
    def enableEffects(self):
        self.apply_effects = True
        self.opacity_effect.setOpacity(1.0)

    def disableEffects(self):
        self.apply_effects = False
        self.opacity_effect.setOpacity(1.0)

    def hasText(self):
        return bool(self.text())

    def hasBackgroundImage(self):
        style_sheet = self.styleSheet()
        return "background-image:" in style_sheet

    def updateEffectsBasedOnStyleAndPixmap(self):
        if self.hasBackgroundImage():
            self.disableEffects()
        else:
            if self.hasText():
                self.enableEffects()
            else:
                self.disableEffects()
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
#  _____________________________________________ VideoPlayer  ____________________________________________________________
class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Manual Feature Extractor")
        self.resize(400, 200)
        self.setWindowIcon(QIcon('imgs/help-circle.svg'))
        self.label = QLabel("Loading Video...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.video_path = video_path
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.video_thread = Thread(target=self.playVideo)
        self.video_thread.start()

    def playVideo(self):
        while True:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(img))
                sleep(0.005)

            cap.release()
#  _____________________________________________ AI_Automated_CAD Screen ____________________________________________________
class AI_Automated_CAD_Screen(QWidget):
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self, main_window):
        super().__init__()
        self.resize(1500, 1000)
        self.setMinimumSize(1500, 1000)
        self.setMaximumSize(1920, 1000)
        
        self.main_window = main_window
        self.main_window.variableChanged2.connect(self.handle_variable_changed)

        self.splash_screen = SplashScreen()
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self)
# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.SecondSetVariable = 1
# _______________________________________ PreProcessing _______________________________________________________ #
   
        self.transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                          ])
        
# ______________________________________________ show_main_content __________________________________________________

        self.show_main_content()

    def show_main_content(self):
        self.image_label = RoundImageLabel()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(600)
        self.image_label.setFixedHeight(600)
        
        self.image_label2 = ImageLabel("AI Automated CAD", apply_effects = False)
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: white;")
        font = QFont()
        font.setPointSize(20)
        font.setBold(False)
        self.image_label2.setFont(font)

        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer2 = QSpacerItem(100, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        HBoxLayout2 = QHBoxLayout()
        HBoxLayout2.addItem(spacer)

        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addItem(spacer2)
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.image_label2)
        VBoxLayout1.addLayout(HBoxLayout2)
        self.setLayout(VBoxLayout1)
# __________________________________________________ Functions _______________________________________________________
    def handle_variable_changed(self, new_value):
        self.SecondSetVariable = new_value
        # print("SecondSetVariable:", self.SecondSetVariable)
        
    def mouseDoubleClickEvent(self, event):
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.dcm *.ima)")
            
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                self.file_path = file_paths[0]
                
                if self.file_path.lower().endswith(('.dcm', '.ima')):
                    dicom_data = pydicom.dcmread(self.file_path)
                    pixel_array = dicom_data.pixel_array
                    image_data_np = pixel_array.astype(np.uint8)
                    self.file_path = self.save_image(image_data_np)
                    
                self.image = cv2.imread(self.file_path)

                if self.image is not None:
                    height, width = self.image.shape[:2]
                    format = QImage.Format_Grayscale8 if len(self.image.shape) == 2 else QImage.Format_RGB888
                    q_img = QImage(self.image.data, width, height, self.image.strides[0], format)

                    pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    rounded_pixmap = self.create_round_image(pixmap)

                    if rounded_pixmap:
                        self.display_image(rounded_pixmap, self.image_label)
                        self.carry_out_Ai_CAD()
                else:
                    print("Error: Unable to load the image.")

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and mime_data.urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.file_path = file_path
        
        if self.file_path.lower().endswith(('.dcm', '.ima')):
            dicom_data = pydicom.dcmread(self.file_path)
            pixel_array = dicom_data.pixel_array
            image_data_np = pixel_array.astype(np.uint8)
            self.file_path = self.save_image(image_data_np)
        
        self.image = cv2.imread(self.file_path)
            
        if self.image is not None:
            height, width = self.image.shape[:2]
            format = QImage.Format_Grayscale8 if len(self.image.shape) == 2 else QImage.Format_RGB888
            q_img = QImage(self.image.data, width, height, self.image.strides[0], format)

            pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            rounded_pixmap = self.create_round_image(pixmap)

            if rounded_pixmap:
                self.display_image(rounded_pixmap, self.image_label)
                self.carry_out_Ai_CAD()
        
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
            
    def save_image(self, image):
        _, temp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_path, image)
        return temp_path
            
    def carry_out_Ai_CAD(self):
        
        if self.SecondSetVariable == 1:
            
            prediction = self.pytorch_classifier(self.file_path, 1).item()
            if prediction == 0:
                self.image_label2.setText(f"Class : Normal")
            if prediction == 1:
                self.image_label2.setText(f"Class : OsteoArthrits")
        
        elif self.SecondSetVariable == 2:
            
            prediction = self.pytorch_classifier(self.file_path, 2).item()
            if prediction == 0:
                self.image_label2.setText(f"Class : Normal")
            if prediction == 1:
                self.image_label2.setText(f"Class : Mild")
            if prediction == 2:
                self.image_label2.setText(f"Class : Severe")
                
        elif self.SecondSetVariable == 3:
            
            prediction = self.pytorch_classifier(self.file_path, 3).item()
            if prediction == 0:
                self.image_label2.setText(f"Class : Normal")
            if prediction == 1:
                self.image_label2.setText(f"Class : Doubtful")
            if prediction == 2:
                self.image_label2.setText(f"Class : Mild")
            if prediction == 3:
                self.image_label2.setText(f"Class : Moderate")
            if prediction == 4:
                self.image_label2.setText(f"Class : Severe")
                        
        else:
            pass

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
        
        return output

    def pytorch_classifier(self, img_path, classification_approach):
        ROI_prediction = self.splash_screen.model(img_path)
        image = cv2.imread(img_path)
        x1,y1,x2,y2 = self.get_coordinate_Predict_image(ROI_prediction)[0][0:4]
        image = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(gray)
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():

            if classification_approach == 1: # for 2 classes approach using cascade fused models 
                outputs = self.splash_screen.pytorch_model_2C(input_batch)
                _, prediction = torch.max(outputs, 1)
                return  prediction
            
            elif classification_approach == 2: # for 3 classes approach
                outputs = self.splash_screen.pytorch_model_3C(input_batch)
                _, prediction = torch.max(outputs, 1)
                return  prediction
            
            elif classification_approach == 3: # for 5 classes approach
                outputs = self.splash_screen.pytorch_model_5C(input_batch)
                _, prediction = torch.max(outputs, 1)
                return  prediction

            else:
                pass

    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label2.setText("AI Automated CAD")

    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
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
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self, main_window):
        super().__init__()
        
        self.resize(1500, 1000)
        self.setMinimumSize(1500, 1000)
        self.setMaximumSize(1920, 1000)
        
        self.main_window = main_window
        self.main_window.variableChanged.connect(self.handle_variable_changed)
         
        self.splash_screen = SplashScreen()
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self)
# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.Features_input = []
        self.FirstSetVariable = 1
# ______________________________________________ show_main_content __________________________________________________
        
        self.show_main_content()

    def show_main_content(self):
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        
        self.image_label = RoundImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(600)
        self.image_label.setFixedHeight(600)
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label2 = ImageLabel("Conventional CAD", apply_effects = False)
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: white;")

        font = QFont()
        font.setPointSize(20)
        font.setBold(False)
        self.image_label2.setFont(font)
        self.setAcceptDrops(True)
        
        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer2 = QSpacerItem(100, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        HBoxLayout2 = QHBoxLayout()
        HBoxLayout2.addItem(spacer)

        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addItem(spacer2)
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.image_label2)
        VBoxLayout1.addLayout(HBoxLayout2)
        self.setLayout(VBoxLayout1)
# __________________________________________________ Functions _______________________________________________________
    def handle_variable_changed(self, new_value):
        self.FirstSetVariable = new_value
        # print("FirstSetVariable:", self.FirstSetVariable)
      
    def handle_variable_changed_Equalized(self, new_value):
        self.perform_intensity_normalization = new_value
         
           
    def mouseDoubleClickEvent(self, event):
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.dcm *.ima *.png *.jpg *.jpeg)")

            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                self.file_path = file_paths[0]
                
                if self.file_path.lower().endswith(('.dcm', '.ima')):
                    dicom_data = pydicom.dcmread(self.file_path)
                    pixel_array = dicom_data.pixel_array
                    image_data_np = pixel_array.astype(np.uint8)
                    self.file_path = self.save_image(image_data_np)
                    
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
    
    
    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and mime_data.urls()[0].isLocalFile():
            event.acceptProposedAction()


    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()

        if file_path.lower().endswith(('.dcm', '.ima')):
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array
            image_data_np = pixel_array.astype(np.uint8)
            file_path = self.save_image(image_data_np)
        
        self.file_path = file_path
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
        
    def save_image(self, image):
        _, temp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_path, image)
        return temp_path
    
    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label2.setText("Conventional CAD")
        self.Features_input = []
        self.set_image(None)
        self.set_path(None)

    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
    
    def set_image(self, img):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image = img
    
    def set_path(self, file_path):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image_path = file_path
        
        
    def carry_out(self):
        self.set_image(self.image)
        self.set_path(self.file_path)
        
        self.Feature_Extraction_and_Visualization_Screen.load_Conventional_Image()
        
        self.Features_input = [[
                               self.Feature_Extraction_and_Visualization_Screen.get_Medial_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Central_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Lateral_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_medial_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_central_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_lateral_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_skewness(),
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['max_probability'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['ASM'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['homogeneity'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['contrast'],
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_features_Normalized()[8],
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_features_Normalized()[0],
                               self.Feature_Extraction_and_Visualization_Screen.get_lbp_features_Normalized()[6],
                               self.Feature_Extraction_and_Visualization_Screen.get_ltp_variance_Normalized(),
                               self.Feature_Extraction_and_Visualization_Screen.get_HOG_Normalized()[2],
                               self.Feature_Extraction_and_Visualization_Screen.get_HOG_Normalized()[6],
                               self.Feature_Extraction_and_Visualization_Screen.get_HOG_Normalized()[5],
                               self.Feature_Extraction_and_Visualization_Screen.get_HOG_Normalized()[3],

        ]]
        
        if self.FirstSetVariable ==1:
            Features_input = self.splash_screen.scaler_Normalize_Binary.transform(self.Features_input)
            Random_Forest_Normale_Binary_Pred = self.splash_screen.Random_Forest_Normale_Binary.predict(Features_input)

            if (Random_Forest_Normale_Binary_Pred == 0):
                self.image_label2.clear()
                self.image_label2.setText(f" Norm")
            
            else:
                self.image_label2.clear()
                self.image_label2.setText(f"Osteo")

        # 3 Class
        elif self.FirstSetVariable == 2: 
            Features_input =self.splash_screen.scaler_Normalize_3_Class.transform(self.Features_input)
            Random_Forest_Normale_3_Class_pred = self.splash_screen.Random_Forest_Normalize_3_Class.predict(Features_input)
            if Random_Forest_Normale_3_Class_pred== 0: 
                self.image_label2.clear()
                self.image_label2.setText(f" Norm,")
            elif Random_Forest_Normale_3_Class_pred== 1: 
                self.image_label2.clear()
                self.image_label2.setText(f" Mild")
            else:
                self.image_label2.clear()
                self.image_label2.setText(f" Severe")

        # 5 Class
        elif self.FirstSetVariable == 3: 
            Features_input =self.splash_screen.scaler_Normalize_5_Class.transform(self.Features_input)
            Random_Forest_Normale_5_Class_pred = self.splash_screen.Random_Forest_Normalize_5_Class.predict(Features_input)

            if Random_Forest_Normale_5_Class_pred== 0: 
                self.image_label2.clear()
                self.image_label2.setText(f" Norm")

            elif Random_Forest_Normale_5_Class_pred== 1: 
                self.image_label2.clear()
                self.image_label2.setText(f"Doubtful")

            elif Random_Forest_Normale_5_Class_pred== 2: 
                self.image_label2.clear()
                self.image_label2.setText(f" Mild")

            elif Random_Forest_Normale_5_Class_pred== 3: 
                self.image_label2.clear()
                self.image_label2.setText(f" Moderate")

            else:
                self.image_label2.clear()

#  Fusion 
        elif self.FirstSetVariable == 4:
                            
            Features_input =self.splash_screen.scaler_0_vs_1.transform(self.Features_input)
            model_0_vs_1_predict = self.splash_screen.model_0_vs_1.predict(Features_input)

            Features_input =self.splash_screen.scaler_0_vs_2.transform(self.Features_input)
            model_0_vs_2_predict = self.splash_screen.model_0_vs_2.predict(Features_input)

            Features_input =self.splash_screen.scaler_0_vs_3.transform(self.Features_input)
            model_0_vs_3_predict = self.splash_screen.model_0_vs_3.predict(Features_input)

            Features_input =self.splash_screen.scaler_0_vs_4.transform(self.Features_input)
            model_0_vs_4_predict = self.splash_screen.model_0_vs_4.predict(Features_input)

            Features_input =self.splash_screen.scaler_1_vs_2.transform(self.Features_input)
            model_1_vs_2_predict = self.splash_screen.model_1_vs_2.predict(Features_input)

            Features_input =self.splash_screen.scaler_1_vs_3.transform(self.Features_input)
            model_1_vs_3_predict = self.splash_screen.model_1_vs_3.predict(Features_input)

            Features_input =self.splash_screen.scaler_2_vs_3.transform(self.Features_input)
            model_2_vs_3_predict = self.splash_screen.model_2_vs_3.predict(Features_input)
 
            Features_input =self.splash_screen.scaler_3_vs_4.transform(self.Features_input)
            model_3_vs_4_predict = self.splash_screen.model_3_vs_4.predict(Features_input)

            prediction = 6

            if model_0_vs_4_predict==4:
                if model_3_vs_4_predict == 4:
                    prediction = 4
                else:
                    if model_2_vs_3_predict == 3:
                        prediction = model_1_vs_3_predict
                    else:
                        if model_1_vs_2_predict == 2:
                            prediction = 2
                        else:
                            if model_0_vs_1_predict == 1:
                                prediction = 1
                                
                            else:
                                prediction = model_0_vs_2_predict


            elif model_0_vs_4_predict ==0:
                if model_0_vs_3_predict   == 3:
                    if model_2_vs_3_predict == 3: 
                        prediction = 3
                    else:
                        if model_1_vs_2_predict == 2:  
                            prediction = 2
                        else: 
                            prediction = model_0_vs_1_predict
                else:
                    if model_0_vs_1_predict == 1:
                        if model_1_vs_2_predict == 2:
                            prediction = model_0_vs_2_predict 
                        else:
                            prediction = model_1_vs_3_predict

                    else:
                        prediction = model_0_vs_2_predict


            if prediction == 0: 
                self.image_label2.clear()
                self.image_label2.setText(f" Normal,")

            elif prediction == 1: 
                self.image_label2.clear()
                self.image_label2.setText(f"Doubtful")

            elif prediction == 2: 
                self.image_label2.clear()
                self.image_label2.setText(f" Mild")

            elif prediction == 3: 
                self.image_label2.clear()
                self.image_label2.setText(f" Moderate")

            else:
                self.image_label2.clear()
                self.image_label2.setText(f" Severe")

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
            pixmap.save("Temp/Saved Figures/histogram.png")
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
            plt.savefig("Temp/Saved Figures/lbp_histogram.png")
            lbp_histogram_image = QImage("Temp/Saved Figures/lbp_histogram.png")
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
            pixmap.save("Temp/Saved Figures/lbp_histogram.png")
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
            plt.savefig("Temp/Saved Figures/ltp_histogram.png")
            ltp_histogram_image = QImage("Temp/Saved Figures/ltp_histogram.png")
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
            pixmap.save("Temp/Saved Figures/ltp_histogram.png")
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
            pixmap1.save("Temp/Saved Figures/lbp_image.png")
            pixmap2.save("Temp/Saved Figures/ltp_image.png")
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
            pixmap1.save("Temp/Saved Figures/Centeral Region Width.png")
            pixmap2.save("Temp/Saved Figures/Tibial Width.png")
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
            pixmap1.save("Temp/Saved Figures/Predicted JSN Original Image.png")
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
            pixmap1.save("Temp/Saved Figures/Histogram of Oriented Gradients (HOG) Window.png")
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
        self.setGeometry(1050, 400, 800, 400)
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
            plt.savefig("Temp/Saved Figures/HOG_Histogram.png")
            HOG_Histogram_img = QImage("Temp/Saved Figures/HOG_Histogram.png")
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
            pixmap.save("Temp/Saved Figures/HOG_Histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
#  _______________________________________________ Settings _____________________________________________________
# class SettingsWindow(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         self.setWindowFlags(Qt.WindowStaysOnTopHint)
#         gradient_style = """
#             background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(64, 64, 64, 75), stop:1 rgba(255, 255, 255, 0));
#         """
#         self.setStyleSheet(gradient_style)
        
#         self.setWindowTitle("Settings")
#         self.setWindowIcon(QIcon("imgs/setting_blacks.svg"))
#         self.setGeometry(150, 400, 800, 400)
        
#         self.Settings_label = QLabel("Settings")
#         self.Settings_label.setStyleSheet("""background-color: transparent;
#                                           color: rgba(255,255,255,200);
#                                           border: none;
#                                           """)
#         self.Settings_label.setAlignment(Qt.AlignCenter)
        
#         font = QFont()
#         font.setPointSize(15)
#         font.setBold(True)
#         self.Settings_label.setFont(font)
        
        
#         self.Settings_label_icon = QLabel()
#         self.Settings_label_icon.setFixedSize(50,50)
#         self.Settings_label_icon.setPixmap(QPixmap("imgs/settings.svg"))
#         self.Settings_label_icon.setStyleSheet("""background-color: transparent;
#                                           border: none;
#                                           text-align: left;
#                                           """)
#         self.Settings_label_icon.setAlignment(Qt.AlignCenter)
        
        
#         layout = QHBoxLayout()
#         layout.setContentsMargins(0,0,0,0)
        
#         layout.addWidget(self.Settings_label_icon, alignment= Qt.AlignCenter)
#         layout.addWidget(self.Settings_label)
#         widget = QWidget()
#         widget.setLayout(layout)
#         widget.setFixedHeight(40)
#         widget.setFixedWidth(800)
#         widget.setStyleSheet("""background-color: transparent;
#                                     color: rgba(64,64,64,100);
#                                     border: none;
#                                     """)


#         layout2 = QHBoxLayout()
#         layout2.setContentsMargins(0,0,0,0)
#         widget2 = QWidget()
#         widget2.setLayout(layout2)
        
        
#         self.inform_label = QLabel("Export your desirable Feature sets to the CSV file")
#         self.inform_label.setStyleSheet("""background-color: transparent;
#                                           color: rgba(255,255,255,200);
#                                           border: none;
#                                           """)
#         self.inform_label.setAlignment(Qt.AlignCenter)
        
#         font = QFont()
#         font.setPointSize(15)
#         font.setBold(False)
#         self.inform_label.setFont(font)
#         layout2.addWidget(self.inform_label)
        
#         self.skewness_checkbox = QCheckBox("skewness")
#         self.skewness_checkbox.setCheckState(True)
#         self.CB_skewness = 1
        
#         self.skewness_checkbox.setStyleSheet("""background-color: transparent;
#                                color" rgba(64,64,64,100);
#                                text-align : left;""")
#         self.skewness_checkbox.stateChanged.connect(self.check_box_skewness)
        
#         centralLayout = QVBoxLayout()
#         centralLayout.setContentsMargins(0,0,0,0)
        
#         centralLayout.addWidget(widget)
#         centralLayout.addWidget(widget2)
        
#         self.setLayout(centralLayout)
        
        
#     def check_box_skewness(self):
#         self.CB_skewness = 0 if (self.skewness_checkbox.isChecked) else  1
#  _________________________________________________ Help ___________________________________________________
class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Help")
        self.setWindowIcon(QIcon("imgs/help-circle.svg"))
        self.setGeometry(150, 400, 800, 400)

        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0,0,0,0)
        widget2 = QWidget()
        widget2.setLayout(layout2)
        
        help1 = """ 
                    Geometrical based Features

                •	Tibial Width (mm): The measurement of the width of the tibia bone in millimeters.
                •	JSN Avg V. Distance (Pixel): Average vertical distance (in pixels) related to joint space narrowing.
                •	JSN Avg V. Distance (mm): Average vertical distance (in millimeters) related to joint space narrowing.
                •	Medial_distance (Pixel): Pixel-based measurement of distance from a reference point to the medial side.
                •	Central_distance (Pixel): Pixel-based measurement of distance from a reference point to the central side.
                •	Lateral_distance (Pixel): Pixel-based measurement of distance from a reference point to the lateral side.
                •	Medial_distance (mm): Measurement of distance from a reference point to the medial side in millimeters.
                •	Central_distance (mm): Measurement of distance from a reference point to the central side in millimeters.
                •	Lateral_distance (mm): Measurement of distance from a reference point to the lateral side in millimeters.
                •	Tibial_Medial_ratio: Ratio of tibial width to medial distance.
                •	Tibial_Central_ratio: Ratio of tibial width to central distance.
                •	Tibial_Lateral_ratio: Ratio of tibial width to lateral distance.
                •	JSN Area (Squared Pixel): Area of joint space narrowing in squared pixels.
                •	JSN Area (Squared mm): Area of joint space narrowing in squared millimeters.
                •	Medial Area (Squared Pixel): Area of joint space narrowing in the medial region in squared pixels.
                •	Central Area (Squared Pixel): Area of joint space narrowing in the central region in squared pixels.
                •	Lateral Area (Squared Pixel): Area of joint space narrowing in the lateral region in squared pixels.
                •	Medial Area (Squared mm): Area of joint space narrowing in the medial region in squared millimeters.
                •	Central Area (Squared mm): Area of joint space narrowing in the central region in squared millimeters.
                •	Lateral Area (Squared mm): Area of joint space narrowing in the lateral region in squared millimeters.
                •	Medial Area (JSN Ratio): Ratio of medial area affected by joint space narrowing.
                •	Central Area (JSN Ratio): Ratio of central area affected by joint space narrowing.
                •	Lateral Area (JSN Ratio): Ratio of lateral area affected by joint space narrowing.
                •	Medial Area Ratio TWPA (%): Percentage of medial area affected by tibial width per area.
                •	Central Area Ratio TWPA (%): Percentage of central area affected by tibial width per area.
                •	Lateral Area Ratio TWPA (%): Percentage of lateral area affected by tibial width per area.

                """
        self.inform_label1 = QLabel(help1)
        self.inform_label1.setStyleSheet("""background-color: white;
                                          color: rgba(0,0,0,255);
                                          border: none;
                                          text-align: left;
                                          """)
        self.inform_label1.setAlignment(Qt.AlignLeft)
        
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(False)
        self.inform_label1.setFont(font)
        self.inform_label1.setContentsMargins(15,15,15,15)
        
        scroll_area1 = QScrollArea(self)
        scroll_area1.setWidgetResizable(True)
        scroll_area1.setWidget(self.inform_label1)
        
        help2 = """
                        
                     Image based Features

                •	Mean: The average pixel intensity value within the image. It gives a sense of the overall brightness of the image.
                •	Sigma (Standard Deviation): A measure of the spread or dispersion of pixel intensity values from the mean. Higher sigma indicates greater variation in pixel intensities.
                •	Skewness: A measure of the asymmetry of the distribution of pixel intensities. Positive skewness indicates a longer tail on the right side of the distribution, while negative skewness indicates a longer tail on the left side.
                •	Kurtosis: A measure of the peakedness or flatness of the distribution of pixel intensities. High kurtosis indicates a sharp peak and heavy tails, while low kurtosis indicates a flatter distribution.
                •	Contrast: Measures the local variations in pixel intensities. Higher contrast indicates larger differences between adjacent pixel values.
                •	Energy: Represents the uniformity or smoothness of the image texture. High energy values indicate more texture variation.
                •	Correlation: Measures the linear relationship between pixel intensities at different locations in the image.
                •	Homogeneity: Reflects the closeness of pixel intensity values in the image. High homogeneity indicates that neighboring pixel intensities are similar.
                •	Dissimilarity: Measures the average absolute difference in pixel intensity values between neighboring pixels.
                •	ASM (Angular Second Moment): A measure of image homogeneity.
                •	Max Probability: The maximum probability of occurrence of a certain texture pattern in the image.
                •	LBP (Local Binary Pattern) features: Descriptors that capture the local texture patterns by comparing each pixel with its neighboring pixels and encoding the result as a binary number.
                •	LTP (Local Ternary Pattern) features:  Descriptors that but capture local ternary patterns.
                •	HOG (Histogram of Oriented Gradients) features: Descriptors that capture the distribution of gradient orientations in different parts of the image.
        
                """

        self.inform_label2 = QLabel(help2)
        self.inform_label2.setStyleSheet("""background-color: white;
                                          color: rgba(0,0,0,255);
                                          border: none;
                                          text-align: left;
                                          """)
        self.inform_label2.setAlignment(Qt.AlignLeft)
        font2 = QFont("Segoe UI")
        font2.setFamily("Arial")
        font2.setPointSize(15)
        font2.setBold(False)
        self.inform_label2.setFont(font2)
        self.inform_label2.setContentsMargins(15,15,15,15)

        scroll_area2 = QScrollArea(self)
        scroll_area2.setWidgetResizable(True)
        scroll_area2.setWidget(self.inform_label2)
            
        manual_layout = QHBoxLayout(self)
        manual_widget = QWidget()
        manual_widget.setLayout(manual_layout)
        manual_widget.setMaximumHeight(90)
        manual_feature_label = QLabel(
            """
            To know how you can manually extract the image-based features
            """
        )

        font = QFont()
        font.setPointSize(14)
        font.setFamily("Helvetica")
        manual_feature_label.setFont(font)
        manual_feature_label.setAlignment(Qt.AlignCenter)
        manual_layout.addWidget(manual_feature_label)

        video_path = "Help/Manual Feature Extractor.mp4"
        watch_video_button = QPushButton("Click here")
        watch_video_button.setCursor(Qt.PointingHandCursor)
        font4 = QFont()
        font4.setPointSize(14)
        watch_video_button.setFont(font4)
        watch_video_button.setStyleSheet("""
                                        border: none;
                                        background-color: transparent;
                                        color: rgb(63, 127, 166);
                                        text-align: left;
                                        """)

        watch_video_button.clicked.connect(lambda: self.openVideoWindow(video_path))
        manual_layout.addWidget(watch_video_button)
        
        spacer2 = QSpacerItem(50, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        manual_layout.addItem(spacer2)
        layout2.addWidget(manual_widget)
        
        layout_horizontal_scroll = QHBoxLayout()
        layout_horizontal_scroll.setContentsMargins(0,0,0,0)
        layout2.addLayout(layout_horizontal_scroll)
        layout_horizontal_scroll.addWidget(scroll_area1)
        layout_horizontal_scroll.addWidget(scroll_area2)
        
        centralLayout = QVBoxLayout()
        centralLayout.setContentsMargins(0,0,0,0)
        centralLayout.addWidget(widget2)
        self.setLayout(centralLayout)
   
    def openVideoWindow(self, video_path):
        video_player = VideoPlayer(video_path)
        video_player.setGeometry(100, 100, 800, 600)
        video_player.show()
#  _____________________________________________ Feature Extraction and Visualization Screen ____________________________________________________________
class Feature_Extraction_and_Visualization_Screen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.resize(1500, 1000)
        self.setMinimumSize(1500, 1000)
        self.setMaximumSize(1920, 1000)
        
        self.main_window = main_window
        self.main_window.variableChanged_Equalize_Feature_Visualization.connect(self.handle_variable_changed_Equalized)
        self.main_window.variableChanged_Auto_mode.connect(self.handle_variable_changed_Auto_mode)
        self.main_window.variableChanged_Visualization.connect(self.handle_variable_changed_Visualization)
           
        self.setMouseTracking(True)
        self.show_main_content()
        
    def show_main_content(self):
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
        
        self.splash_screen = SplashScreen()
        
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
        self.JSN_label = ImageLabel(self, apply_effects = False)
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.JSN_label, image_path)
        self.JSN_label.setAlignment(Qt.AlignCenter)
        self.JSN_label.setScaledContents(True)
        self.JSN_label.setMouseTracking(True)
        self.JSN_label.mouseMoveEvent = self.showToolTip
        self.YOLO_label = ImageLabel("ROI Detection")
        self.YOLO_label.setAlignment(Qt.AlignCenter)
        self.YOLO_label.setScaledContents(True)
        self.YOLO_label.setMouseTracking(True)
        self.YOLO_label.mouseMoveEvent = self.showToolTip_ROI_Detector
        self.Intensity_label = ImageLabel("Equalization")
        self.Intensity_label.setAlignment(Qt.AlignCenter)
        self.Intensity_label.setStyleSheet("color: white;")
        self.Binarization_label = ImageLabel("Binarization")
        self.Binarization_label.setAlignment(Qt.AlignCenter)
        self.Binarization_label.setStyleSheet("color: white;")
        self.edge_label = ImageLabel("Edge Detection")
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
        
        self.image_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        # self.JSN_label.setStyleSheet(" border-radius: 50px; background-image: url(imgs/Feature_Extraction.png); background-repeat: no-repeat; background-color: rgba(0, 0, 0, 0); color: darkgray;")
        self.YOLO_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.Intensity_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.Binarization_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.edge_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")

        self.image_label.updateEffectsBasedOnStyleAndPixmap()        
        self.JSN_label.updateEffectsBasedOnStyleAndPixmap()        
        self.YOLO_label.updateEffectsBasedOnStyleAndPixmap()        
        self.Intensity_label.updateEffectsBasedOnStyleAndPixmap()        
        self.Binarization_label.updateEffectsBasedOnStyleAndPixmap()        
        self.edge_label.updateEffectsBasedOnStyleAndPixmap()        

        layoutH2_space1 = QHBoxLayout()
        layoutH2_space1_Widget = QWidget()
        layoutH2_space1_Widget.setFixedHeight(20)
        layoutH2_space1_Widget.setLayout(layoutH2_space1)
        layoutH2_space1_Widget.setStyleSheet("""
                                             background-color: transparent;
                                             border: none;
                                             """)
        
        layoutH2_space2 = QHBoxLayout()
        layoutH2_space2_Widget = QWidget()
        layoutH2_space2_Widget.setFixedHeight(50)
        layoutH2_space2_Widget.setLayout(layoutH2_space2)
        layoutH2_space2_Widget.setStyleSheet("""
                                             background-color: transparent;
                                             border: none;
                                             """)
        
        
        layout = QVBoxLayout()
        layoutH1 = QHBoxLayout()
        layout.addWidget(layoutH2_space1_Widget)
        
        layout.addLayout(layoutH1)
        layoutH1.addWidget(self.image_label)
        layoutH1.addWidget(self.JSN_label)
        layoutH1.addWidget(self.YOLO_label)
        layoutH = QHBoxLayout()
        layout.addLayout(layoutH)
  
        layout.addWidget(layoutH2_space2_Widget)

        layoutH.addWidget(self.Intensity_label)
        layoutH.addWidget(self.Binarization_label)
        layoutH.addWidget(self.edge_label)
        
        self.setLayout(layout)
        self.histogram_window = HistogramWindow()
        self.histogram_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.histogram_window.set_histogram(None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_histogram_display)
        self.timer.start(1)
# _____________________________________________ Iitialization _______________________________________________
        self.folder_name = "Temp"
        self.mouse_double_clicked = 0
        self.switch = True
        self.switch2 = False
        self.Equalize_button_switch = True
        self.mode = True        
        self.image = None
        self.image_path = None
        self.Conventional_image = None
        self.Conventional_image_path = None
        self.cropped_images = []
        self.user_cropped = False        
        self.setAcceptDrops(True)
        self.center_crop = ()
        self.perform_intensity_normalization = True
        self.perform_canny_edge = True
        self.end_Automation = 0
        self.conventional_image_Indicator = 0
        self.check_knee_literality = False
        self.percentage = 0
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
        self.lbp_features_Normalized = []
        self.lbp_variance_Normalized = 0
        self.lbp_entropy_Normalized = 0
        self.ltp_features = []
        self.ltp_variance = 0
        self.ltp_entropy = 0
        self.ltp_features_Normalized = []
        self.ltp_variance_Normalized = 0
        self.ltp_entropy_Normalized = 0
        self.hog_bins = []
        self.hog_bins_Normalized = []
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
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")

    def handle_variable_changed_Equalized(self, new_value):
        self.perform_intensity_normalization = new_value
        
    def handle_variable_changed_Auto_mode(self, new_value):
        self.switch = new_value
    
    def handle_variable_changed_Visualization(self, new_value):
        self.switch2 = new_value
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        image_path = "imgs/Feature_Extraction.png"
        
        if self.mouse_double_clicked == 1:
            pass
        else:
            self.set_background_image(self.JSN_label, image_path)
        
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
        if self.YOLO_label.text() == "ROI Detection":
            pass
        else:
            if self.prob is not None:
                QToolTip.showText(event.globalPos(), f"Probability is {self.prob}")
            else:
                QToolTip.showText(event.globalPos(), "")
        
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
        
        # y = int(self.Conventional_image.shape[0] * 0.5)
        # x = int(self.Conventional_image.shape[1] * 0.5)
        # self.center_crop = (x, y)
        # crop_size = 224
        # x1 = x - crop_size // 2
        # x2 = x + crop_size // 2
        # y1 = y - crop_size // 2
        # y2 = y + crop_size // 2
        # x1 = max(x1, 0)
        # x2 = min(x2, self.Conventional_image.shape[1])
        # y1 = max(y1, 0)
        # y2 = min(y2, self.Conventional_image.shape[0])
        # self.Conventional_image = self.Conventional_image[y1:y2, x1:x2]
        
        
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
        
        if self.JSN_Area_Total != 0:
            self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
            self.central_area_Ratio = self.central_area / self.JSN_Area_Total
            self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
            
            print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
            print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
            print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
        if self.Tibial_width_Predicted_Area != 0:
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
        self.save_PNGs("Temp/Saved Figures")
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        self.on_button_click()

    def load_image(self, image_path):
        self.user_cropped = False
        self.image_path = image_path
        self.image = cv2.imread(image_path)
            
        # y = int(self.image.shape[0] * 0.5)
        # x = int(self.image.shape[1] * 0.5)
        # self.center_crop = (x, y)
        # crop_size = 224
        # x1 = x - crop_size // 2
        # x2 = x + crop_size // 2
        # y1 = y - crop_size // 2
        # y2 = y + crop_size // 2
        # x1 = max(x1, 0)
        # x2 = min(x2, self.image.shape[1])
        # y1 = max(y1, 0)
        # y2 = min(y2, self.image.shape[0])
        # self.image = self.image[y1:y2, x1:x2]

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
            if self.JSN_Area_Total != 0:
                self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
                
                print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
                print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
                print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
                
                
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            if self.Tibial_width_Predicted_Area != 0:
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
            self.main_window.toggleMinimized()
            # self.print_to_cmd("\033[92mDon't Panic, It'll restore its screen again after the automation ends.\033[0m")
            # self.print_to_cmd("\033[92mIf you want to interrupt exit the application from windows Taskbar.\033[0m")

            self.current = image_path
            directory_path = os.path.dirname(image_path)
            
            if image_path.lower().endswith(('.dcm')):
                image_files = glob.glob(os.path.join(directory_path, "*.dcm"))
            
            elif image_path.lower().endswith(('.ima')):
                image_files = glob.glob(os.path.join(directory_path, "*.ima"))
                
            elif image_path.lower().endswith(('.jpeg')):
                image_files = glob.glob(os.path.join(directory_path, "*.jpeg"))
            
            elif image_path.lower().endswith(('.jpg')):
                image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
                
            elif image_path.lower().endswith(('.png')):
                image_files = glob.glob(os.path.join(directory_path, "*.png"))
                self.total_image_files = len(image_files)
            else:
                print("\033[91mUnsupported image format.\033[0m")
                pass
            
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
                if self.JSN_Area_Total != 0:
            
                    self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                    self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                    self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total

                    print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
                    print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
                    print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
                if self.Tibial_width_Predicted_Area != 0:
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
        
                # self.print_to_cmd(f"\033[92mImg No {counter} out of {self.total_image_files} image files.\033[0m")
                
                # print(f"\033[92myolo_results = {yolo_results}.\033[0m")
                print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
    
            self.main_window.toggleMinimized()
            end_time = time.time()
            self.overall_time = end_time - start_time
            hours = self.overall_time // 3600
            remaining_time = self.overall_time % 3600
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            
            print(f"\033[92mTotal Computational Time = {hours} Hours, {minutes} Minutes, and {seconds} Seconds.\033[0m")
            # self.print_to_cmd(f"\033[92mTotal Computational Time = {hours} Hours, {minutes} Minutes, and {seconds} Seconds.\033[0m")
            
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
            if self.JSN_Area_Total != 0:
                self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
                
                print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
                print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
                print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            if self.Tibial_width_Predicted_Area != 0:
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
            self.save_PNGs("Temp/Saved Figures")
            img = cv2.imread("Temp/Saved Figures/JSN-Region.png")
            self.perform_edge_detection(img)

        self.calculate_histogram()
        self.save_PNGs("Temp/Saved Figures")

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
                
                img_bytes = img.tobytes()
                q_img = QImage(img_bytes, width, height, bytes_per_line, format)
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
                               
                if self.Tibial_width != 0:
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

# ______________________________  Vertical Distance and Area _________________________________ #

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

                            # get down point to get vertical distance 
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
                
                if distace_Central == []:
                    pass
                
                else:
                    distace_Central.pop(0)

                self.average_medial_distance =  np.mean(distace_Left)
                self.average_central_distance =  np.mean (distace_Central)
                self.average_lateral_distance = np.mean(distace_Right)
                
                self.average_distance = (self.average_medial_distance + self.average_central_distance + self.average_lateral_distance) / 3
                self.average_distance_mm = self.average_distance / self.length_ratio
                
                self.medial_area = np.sum(distace_Left)
                self.central_area = np.sum(distace_Central)
                self.lateral_area = np.sum(distace_Right)

                self.JSN_Area_Total =  self.medial_area + self.central_area + self.lateral_area
                self.JSN_Area_Total_Squared_mm = self.JSN_Area_Total / self.area_ratio
#__________________________________________  end ______________________________________________#
                
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

    
    def Save_ALL(self):
        save_directory = QFileDialog.getExistingDirectory(None, "Select Directory to Save Files", QDir.homePath())

        if not save_directory:
            return
        
        save_as_dcm = self.ask_save_as_dcm_dialog()

        text = "All figures have been saved successfully.\n"
        msgBox = CustomMessageBox2()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowIcon(QIcon("imgs/save.svg"))
        msgBox.setWindowTitle("save")
        msgBox.setText(text)
        msgBox.addButton(QMessageBox.Ok)
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msgBox.exec_()
        
        if save_as_dcm:
            self.save_PNGs_as_DICOM(save_directory)
        else:
            self.save_PNGs(save_directory)
        
        
    def ask_save_as_dcm_dialog(self):
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle("Save As")
        dialog.setText("Do you want to save as DICOM or PNG?")
        dialog.addButton("DICOM", QMessageBox.YesRole)
        dialog.addButton("PNG", QMessageBox.NoRole)
        result = dialog.exec_()

        return result == 0
    
    def save_PNGs(self, save_directory):
        labels = [
            (self.image_label, "Manual Feature Extractor image.png"),
            (self.JSN_label, "JSN_label_Sub-Regions.png"),
            (self.YOLO_label, "YOLO_ROI.png"),
            (self.Intensity_label, "Intensity_label.png"),
            (self.Binarization_label, "Binarization_label.png"),
            (self.edge_label, "canny-edge-label.png"),
            (self.PredictedJSNOriginalWindow.Predicted_JSN_label, "JSN-Region.png"),
            (self.HistogramOfOrientedGradientsImage.HOG_image_label, "HOG Image.png"),
            (self.HOGHistogram.HOGHistogram_label, "HOG_Histogram.png"),
            (self.LBPandLTPImgsWindow.LBPImgWindow_label, "lbp_image.png"),
            (self.LBPandLTPImgsWindow.LTPImgWindow_label, "ltp_image.png"),
            (self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label, "Tibial Width.png"),
            (self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label, "Centeral Region Width.png"),
            (self.histogram_window.Histogram_Intensity_label, "histogram.png"),
            (self.LBPHistogramWindow.LBPHistogramWindow_label, "lbp_histogram.png"),
            (self.LTPHistogramWindow.LTPHistogramWindow_label, "ltp_histogram.png")
            
        ]

        for label, filename in labels:
            filepath = os.path.join(save_directory, filename)
            pixmap = label.grab()
            img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            resized_image = cv2.resize(image, (224, 224))
            cv2.imwrite(filepath, resized_image)

        save_directory = None
        
            
    def save_PNGs_as_DICOM(self, save_directory):

        labels = [
            (self.image_label, "Manual Feature Extractor image.png"),
            (self.JSN_label, "JSN_label_Sub-Regions.png"),
            (self.YOLO_label, "YOLO_ROI.png"),
            (self.Intensity_label, "Intensity_label.png"),
            (self.Binarization_label, "Binarization_label.png"),
            (self.edge_label, "canny-edge-label.png"),
            (self.PredictedJSNOriginalWindow.Predicted_JSN_label, "JSN-Region.png"),
            (self.HistogramOfOrientedGradientsImage.HOG_image_label, "HOG Image.png"),
            (self.HOGHistogram.HOGHistogram_label, "HOG_Histogram.png"),
            (self.LBPandLTPImgsWindow.LBPImgWindow_label, "lbp_image.png"),
            (self.LBPandLTPImgsWindow.LTPImgWindow_label, "ltp_image.png"),
            (self.CenteralRegandTibialWidthImgsWindow.TibImgWindow_label, "Tibial Width.png"),
            (self.CenteralRegandTibialWidthImgsWindow.CRegImgWindow_label, "Centeral Region Width.png"),
            (self.histogram_window.Histogram_Intensity_label, "histogram.png"),
            (self.LBPHistogramWindow.LBPHistogramWindow_label, "lbp_histogram.png"),
            (self.LTPHistogramWindow.LTPHistogramWindow_label, "ltp_histogram.png")
            
        ]
        
        for label, filename in labels:
            filepath = os.path.join(save_directory, filename)
            pixmap = label.grab()
            img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            image = cv2.resize(image, (224, 224))
            dicom_dataset_rgb = self.convert_image_to_dicom(image)
            dicom_filepath_rgb = filepath.replace(".png", "_rgb.dcm")
            dicom_dataset_rgb.save_as(dicom_filepath_rgb, write_like_original=False)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dicom_dataset_gray = self.convert_image_to_dicom(image_gray)
            dicom_dataset_rgb.PixelData = dicom_dataset_gray.PixelData
            dicom_filepath_gray = filepath.replace(".png", "_gray.dcm")
            dicom_dataset_rgb.save_as(dicom_filepath_gray, write_like_original=False)
            os.remove(dicom_filepath_rgb)

        save_directory = None
    

    def convert_image_to_dicom(self, image):
        dicom_dataset = FileDataset(None, {}, preamble=b"\0" * 128)

        if len(image.shape) == 2:
            dicom_dataset.Rows, dicom_dataset.Columns = image.shape
        elif len(image.shape) == 3:
            dicom_dataset.Rows, dicom_dataset.Columns, dicom_dataset.Frames = image.shape
        else:
            raise ValueError("Unsupported image dimensions")

        dicom_dataset.SamplesPerPixel = 1
        dicom_dataset.BitsAllocated = 8
        dicom_dataset.BitsStored = 8
        dicom_dataset.HighBit = 7
        dicom_dataset.PixelRepresentation = 0
        dicom_dataset.PhotometricInterpretation = "MONOCHROME2"

        dicom_dataset.PixelSpacing = [1.0, 1.0]
        dicom_dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
        dicom_dataset.RescaleIntercept = 0
        dicom_dataset.RescaleSlope = 1

        image = image.astype(np.uint8)
        pixel_data = image.flatten()
        dicom_dataset.PixelData = pixel_data.tobytes()
        dicom_dataset.WindowCenter = 0
        dicom_dataset.WindowWidth = np.max(image) - np.min(image)
        dicom_dataset.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # X-Ray Image Storage
        dicom_dataset.Modality = "DX"

        dicom_dataset.SOPInstanceUID = generate_uid()
        dicom_dataset.file_meta = Dataset()
        dicom_dataset.file_meta.MediaStorageSOPInstanceUID = dicom_dataset.SOPInstanceUID
        dicom_dataset.file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # X-Ray Image Storage
        dicom_dataset.file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')  # Explicit VR Little Endian
        dicom_dataset.file_meta.ImplementationClassUID = UID('1.2.826.0.1.3680043.9.7461.1')

        return dicom_dataset
    #  _____________________________ setter ____________________________________  #
    
    def set_image(self, image):
        self.Conventional_image = image
    
    def set_path(self, path):
        self.Conventional_image_path = path
                
    #  _____________________________ getter ____________________________________  #
              
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
    
    def get_medial_area_Ratio_TWPA (self):
        return self.medial_area_Ratio_TWPA 
    
    def get_central_area_Ratio_TWPA (self):
        return self.central_area_Ratio_TWPA 
    
    def get_lateral_area_Ratio_TWPA (self):
        return self.lateral_area_Ratio_TWPA 
    
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
    
    def get_lbp_features_Normalized(self):
        return self.lbp_features_Normalized
    
    def get_lbp_variance_Normalized(self):
        return self.lbp_variance_Normalized
    
    def get_lbp_entropy_Normalized(self):
        return self.lbp_entropy_Normalized
    
    def get_ltp_features(self):
        return self.ltp_features
        
    def get_ltp_variance(self):
        return self.ltp_variance
    
    def get_ltp_entropy(self):
        return self.ltp_entropy
    
    def get_ltp_features_Normalized(self):
        return self.ltp_features_Normalized
        
    def get_ltp_variance_Normalized(self):
        return self.ltp_variance_Normalized
    
    def get_ltp_entropy_Normalized(self):
        return self.ltp_entropy_Normalized
    
    def get_HOG(self):
        return self.hog_bins
    
    def get_HOG_Normalized(self):
        return self.hog_bins_Normalized
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
                            
    def dragEnterEvent(self, event):
        self.image_label.disableEffects()        
        self.JSN_label.disableEffects()        
        self.YOLO_label.disableEffects()        
        self.Intensity_label.disableEffects()        
        self.Binarization_label.disableEffects()        
        self.edge_label.disableEffects() 
        
        self.mouse_double_clicked = 1
        
        mime_data = event.mimeData()
        if mime_data.hasUrls() and mime_data.urls()[0].isLocalFile():
            event.acceptProposedAction()


    def dropEvent(self, event):
        image_path = event.mimeData().urls()[0].toLocalFile()

        if image_path.lower().endswith(('.dcm', '.ima')):
                    dicom_data = pydicom.dcmread(image_path)
                    pixel_array = dicom_data.pixel_array
                    image_data_np = pixel_array.astype(np.uint8)
                    image_path  = self.save_image(image_data_np)
                    
        self.load_image(image_path)

    def mouseDoubleClickEvent(self, event):
        self.image_label.disableEffects()        
        self.JSN_label.disableEffects()        
        self.YOLO_label.disableEffects()        
        self.Intensity_label.disableEffects()        
        self.Binarization_label.disableEffects()        
        self.edge_label.disableEffects()   

        self.mouse_double_clicked = 1

        file_dialog = QFileDialog()
        file_dialog.setNameFilter("DICOM Images (*.dcm *ima);; PNG Images (*.png *.jpeg *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                image_path = file_paths[0]
                
                if image_path.lower().endswith(('.dcm', '.ima')):
                    dicom_data = pydicom.dcmread(image_path)
                    pixel_array = dicom_data.pixel_array
                    image_data_np = pixel_array.astype(np.uint8)
                    image_path  = self.save_image(image_data_np)
                    
                self.load_image(image_path)

    def save_image(self, image):
        _, temp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_path, image)
        return temp_path

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
            self.mouse_double_clicked = 0
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
            image_path = "imgs/Feature_Extraction.png"
            self.set_background_image(self.JSN_label, image_path)
            self.YOLO_label.setText("ROI Detection")
            self.Intensity_label.setText("Equalization")
            self.Binarization_label.setText("Binarization")
            self.edge_label.setText("Edge Detection")
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
    
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
      
    def toggle_intensity_normalization(self):
        if self.perform_intensity_normalization == False:
            self.perform_intensity_normalization = True
            
            self.Equalize_button.setIcon(QIcon('imgs/check-square.svg'))
            self.Equalize_button.setMinimumSize(40,40)
            self.Equalize_button.setMaximumSize(40,40)
            self.Equalize_button.setIconSize(QSize(40,40))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        else:

            self.perform_intensity_normalization = False

            self.Equalize_button.setIcon(QIcon('imgs/square.svg'))
            self.Equalize_button.setMinimumSize(40,40)
            self.Equalize_button.setMaximumSize(40,40)
            self.Equalize_button.setIconSize(QSize(40,40))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

            self.Intensity_label.clear()
            self.Binarization_label.clear()
            self.LBPandLTPImgsWindow.LBPImgWindow_label.clear()
            self.LBPandLTPImgsWindow.LTPImgWindow_label.clear()
            self.LBPHistogramWindow.LBPHistogramWindow_label.clear()
            self.LTPHistogramWindow.LTPHistogramWindow_label.clear()
            self.HistogramOfOrientedGradientsImage.HOG_image_label.clear()
            self.HOGHistogram.HOGHistogram_label.clear()
            self.Binarization_label.setText("Binarization")
            self.Intensity_label.setText("Equalization")
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

    
    def calculate_cooccurrence_parameters_YOLO(self, image):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    
    def calculate_lbp_features_YOLO(self, image):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            radius = 1
            n_points = 8 * radius
            lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
            lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            lbp_histogram_Normalized = lbp_histogram / (lbp_histogram.sum() + 1e-5)
            # lbp_histogram = lbp_histogram / (lbp_histogram.sum() + 1e-5)
            self.LBPHistogramWindow.set_lbp_histogram(lbp_histogram)
            return lbp_histogram, lbp_image, lbp_histogram_Normalized

    def calculate_ltp_features_YOLO(self, image, num_bins=10):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
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
            ltp_histogram_Normalized = ltp_histogram / (ltp_histogram.sum() + 1e-5)
            # ltp_histogram = ltp_histogram / (ltp_histogram.sum() + 1e-5)
            self.LTPHistogramWindow.set_LTPHistogramWindow_histogram(ltp_histogram)
            return ltp_histogram, ltp_image, ltp_histogram_Normalized

    def compute_hog_features(self, image, cell_size=(8, 8), block_size=(1, 1), bins=9):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
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
            # hog_bins /= (np.linalg.norm(hog_bins) + epsilon)
            hog_bins_Normalized = hog_bins /  (np.linalg.norm(hog_bins) + epsilon)
            
            # print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            # print(f"\033[92mlength of Histogram Of Oriented Gradients:{len(hog_bins)} bins\n\033[0m")
            # print(f"\033[92mHOG Feature vector:{(hog_bins)}\n\033[0m")
            # print(f"\033[92mHOG Feature vector (Normalized):{(hog_bins_Normalized)}\n\033[0m")
            # print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")


            self.HOGHistogram.set_HOG_Histogram(hog_bins)
            self.HistogramOfOrientedGradientsImage.set_HOG_Image(hog_image)
            
            return hog_bins, hog_bins_Normalized

        else:
            return []

            
    def calculate_and_save_all_parameters(self, cropped_image, image_path,top_left,bottom_right):
            if cropped_image is not None:
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                self.cooccurrence_properties = self.calculate_cooccurrence_parameters_YOLO(cropped_image_gray)
                self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(cropped_image_gray)
                self.lbp_features, self.lbp_image, self.lbp_features_Normalized = self.calculate_lbp_features_YOLO(cropped_image_gray)
                self.ltp_features, self.ltp_image, self.ltp_features_Normalized = self.calculate_ltp_features_YOLO(cropped_image_gray)               
                self.LBPandLTPImgsWindow.set_lbp_image(self.lbp_image)
                self.LBPandLTPImgsWindow.set_ltp_image(self.ltp_image)
                self.lbp_variance = np.var(self.lbp_features)
                self.ltp_variance = np.var(self.ltp_features)
                self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                
                self.lbp_variance_Normalized = np.var(self.lbp_features_Normalized)
                self.ltp_variance_Normalized = np.var(self.ltp_features_Normalized)
                self.lbp_entropy_Normalized = -np.sum(self.lbp_features_Normalized * np.log2(self.lbp_features_Normalized + 1e-5))
                self.ltp_entropy_Normalized = -np.sum(self.ltp_features_Normalized * np.log2(self.ltp_features_Normalized + 1e-5))
                
                self.hog_bins, self.hog_bins_Normalized = self.compute_hog_features(cropped_image_gray)
                # self.hog_bins, hog_bins_Normalized = self.compute_hog_features(img)
                        
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                csv_file_path = "Temp/CSV/Manual_statistics.csv"
                write_header = not os.path.exists(csv_file_path)
                try:
                    with open(csv_file_path, mode="a", newline="") as csv_file:
                        fieldnames = ["Image Name", "Class","Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                    [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"]+ \
                                    [f"LBP_Normalized_{i}" for i in range(len(self.lbp_features_Normalized))] + ["LBP_Variance_Normalized", "LBP_Entropy_Normalized"] + \
                                    [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"] +\
                                    [f"LTP_Normalized_{i}" for i in range(len(self.ltp_features_Normalized))] + ["LTP_Variance_Normalized", "LTP_Entropy_Normalized"] + \
                                    [f"HOG_{i}" for i in range(len(self.hog_bins))] + \
                                    [f"HOG_Normalized_{i}" for i in range(len(self.hog_bins_Normalized))]
                                    
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
                            **{f"LBP_Normalized_{i}": self.lbp_features_Normalized[i] for i in range(len(self.lbp_features_Normalized))},
                            "LBP_Variance_Normalized": self.lbp_variance_Normalized,
                            "LBP_Entropy_Normalized": self.lbp_entropy_Normalized,
                            **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                            "LTP_Variance": self.ltp_variance,
                            "LTP_Entropy": self.ltp_entropy,
                            **{f"LTP_Normalized_{i}": self.ltp_features_Normalized[i] for i in range(len(self.ltp_features_Normalized))},
                            "LTP_Variance_Normalized": self.ltp_variance_Normalized,
                            "LTP_Entropy_Normalized": self.ltp_entropy_Normalized,
                            **{f"HOG_{i}": self.hog_bins[i] for i in range(len(self.hog_bins))},
                            **{f"HOG_Normalized_{i}": self.hog_bins_Normalized[i] for i in range(len(self.hog_bins_Normalized))}
                            
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
                        self.lbp_features, self.lbp_image, self.lbp_features_Normalized = self.calculate_lbp_features_YOLO(roi_cropped_image)
                        self.ltp_features, self.ltp_image, self.ltp_features_Normalized = self.calculate_ltp_features_YOLO(roi_cropped_image)
                        self.LBPandLTPImgsWindow.set_lbp_image(self.lbp_image)
                        self.LBPandLTPImgsWindow.set_ltp_image(self.ltp_image)
                        self.lbp_variance = np.var(self.lbp_features)
                        self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                        self.lbp_variance_Normalized = np.var(self.lbp_features_Normalized)
                        self.lbp_entropy_Normalized = -np.sum(self.lbp_features_Normalized * np.log2(self.lbp_features_Normalized + 1e-5))
                        
                        if self.ltp_features is not None:
                            self.ltp_variance = np.var(self.ltp_features)
                            self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                        else:
                            self.ltp_variance = None
                            self.ltp_entropy = None
                        
                        
                        if self.ltp_features_Normalized is not None:
                            self.ltp_variance_Normalized = np.var(self.ltp_features_Normalized)
                            self.ltp_entropy_Normalized = -np.sum(self.ltp_features_Normalized * np.log2(self.ltp_features_Normalized + 1e-5))
                        else:
                            self.ltp_variance_Normalized = None
                            self.ltp_entropy_Normalized = None
                            
                        self.hog_bins, self.hog_bins_Normalized = self.compute_hog_features(roi_cropped_image)
                        # self.hog_bins, hog_bins_Normalized = self.compute_hog_features(img)
                        
                        file_name = os.path.splitext(os.path.basename(image_path))[0]
                        csv_file_path_yolo = "Temp/CSV/Automatic_statistics.csv"
                        write_header = not os.path.exists(csv_file_path_yolo)
                        try:
                            with open(csv_file_path_yolo, mode="a", newline="") as csv_file:
                                fieldnames = ["Image Name", "Class", "percentage", "Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "YOLO_Probability", "Tibial Width (Pixel)", "Tibial Width (mm)", "JSN Avg V.Distance (Pixel)", "JSN Avg V.Distance (mm)", "Medial_distance (Pixel)", "Central_distance (Pixel)", "Lateral_distance (Pixel)","Medial_distance (mm)", "Central_distance (mm)", "Lateral_distance (mm)", "Tibial_Medial_ratio", "Tibial_Central_ratio", "Tibial_Lateral_ratio", "JSN Area (Squared Pixel)", "JSN Area (Squared mm)", "Medial Area (Squared Pixel)", "Central Area (Squared Pixel)", "Lateral Area (Squared Pixel)","Medial Area (Squared mm)", "Central Area (Squared mm)", "Lateral Area (Squared mm)", "Medial Area (JSN Ratio)", "Central Area (JSN Ratio)", "Lateral Area (JSN Ratio)", "Medial Area Ratio TWPA (%)","Central Area Ratio TWPA (%)", "Lateral Area Ratio TWPA (%)","Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                            [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"] + \
                                            [f"LBP_Normalized_{i}" for i in range(len(self.lbp_features_Normalized))] + ["LBP_Variance_Normalized", "LBP_Entropy_Normalized"] + \
                                            [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"] + \
                                            [f"LTP_Normalized_{i}" for i in range(len(self.ltp_features_Normalized))] + ["LTP_Variance_Normalized", "LTP_Entropy_Normalized"] + \
                                            [f"HOG_{i}" for i in range(len(self.hog_bins))] + \
                                            [f"HOG_Normalized_{i}" for i in range(len(self.hog_bins_Normalized))]
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
                                    **{f"LBP_Normalized_{i}": self.lbp_features_Normalized[i] for i in range(len(self.lbp_features_Normalized))},
                                    "LBP_Variance_Normalized": self.lbp_variance_Normalized,
                                    "LBP_Entropy_Normalized": self.lbp_entropy_Normalized,
                                    **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                                    "LTP_Variance": self.ltp_variance,
                                    "LTP_Entropy": self.ltp_entropy,
                                    **{f"LTP_Normalized_{i}": self.ltp_features_Normalized[i] for i in range(len(self.ltp_features_Normalized))},
                                    "LTP_Variance_Normalized": self.ltp_variance_Normalized,
                                    "LTP_Entropy_Normalized": self.ltp_entropy_Normalized,
                                    **{f"HOG_{i}": self.hog_bins[i] for i in range(len(self.hog_bins))},
                                    **{f"HOG_Normalized_{i}": self.hog_bins_Normalized[i] for i in range(len(self.hog_bins_Normalized))}
                                    
                                }
                                writer.writerow(row_data)
                        except Exception as e:
                            print("\033[91mError writing to statistics_YOLO.csv:\033[0m", str(e))
#  ________________________________________________ HomeScreen _______________________________________________________
class HomeScreen(QMainWindow):
    variableChanged = pyqtSignal(int)
    variableChanged2 = pyqtSignal(int)
    
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.resize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        self.setMaximumSize(1920, 1080)
# _____________________________________________ Iitialization _______________________________________________
        self.perform_intensity_normalization = True
        self.switch = True
        self.switch2 = False
        self.FirstSetVariable = 1
        self.SecondSetVariable = 1
        self.folder_name = "Temp"

        folder_path = Path(self.folder_name)
        if not folder_path.is_dir():
            folder_path.mkdir()
            print(f"Folder '{self.folder_name}' created successfully.")
        else:
            print(f"Folder '{self.folder_name}' already exists.")
            
        self.folder_name2 = "Temp/CSV"
        folder_path2 = Path(self.folder_name2)
        if not folder_path2.is_dir():
            folder_path2.mkdir()
            print(f"Folder '{self.folder_name2}' created successfully.")
        else:
            print(f"Folder '{self.folder_name2}' already exists.")
        
        self.folder_name3 = "Temp/Saved Figures"
        folder_path3 = Path(self.folder_name3)
        if not folder_path3.is_dir():
            folder_path3.mkdir()
            print(f"Folder '{self.folder_name3}' created successfully.")
        else:
            print(f"Folder '{self.folder_name3}' already exists.")

# _____________________________________________ Classes instances _______________________________________________
        
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self)
        self.Conventional_CAD_Screen = Conventional_CAD_Screen(self)
        self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(self)
        self.HelpWindow = HelpWindow()
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.isFullScreen = True
        # self.showFullScreen()
                
        self.splash_screen = SplashScreen(self)
        self.setCentralWidget(self.splash_screen)
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.show_main_content)
        self.loading_timer.start(500)    
        
        self.background_label = QLabel(self)
        self.pixmap = QPixmap("imgs/33blackSplashScreen.png")
        self.background_label.setPixmap(self.pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Align to top-left
        
        self.central_widget = QWidget(self)
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0,0,0,0)
        
        self.nav_layout_reduced = QVBoxLayout()
        self.nav_layout_reduced.setContentsMargins(0,0,0,0)
        
        self.nav_layout = QVBoxLayout()
        self.nav_layout.setContentsMargins(0,0,0,0)
    
        
    def show_main_content(self):
        self.loading_timer.stop()
        self.splash_screen.hide()       
        
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)

        font = QFont()
        font.setPointSize(16)
               
        self.Feature_Extraction_and_Visualization_button = QPushButton(self)
        self.Feature_Extraction_and_Visualization_button.setIcon(QIcon('imgs/Feature_Extraction_button.png'))
        self.Feature_Extraction_and_Visualization_button.setMinimumSize(420,70)
        self.Feature_Extraction_and_Visualization_button.setIconSize(QSize(45,45))
        self.Feature_Extraction_and_Visualization_button.setText(" Feature Extraction and Visualization")
        self.Feature_Extraction_and_Visualization_button.setFont(font)
        self.Feature_Extraction_and_Visualization_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.Feature_Extraction_and_Visualization_button_layout = QHBoxLayout()
        self.Feature_Extraction_and_Visualization_button_layout.setContentsMargins(10,0,0,0)
        self.Feature_Extraction_and_Visualization_button_layout.addWidget(self.Feature_Extraction_and_Visualization_button, alignment= Qt.AlignLeft)
        self.Feature_Extraction_and_Visualization_button_layout_Widget = QWidget(self)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setFixedWidth(420)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setFixedHeight(70)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setLayout(self.Feature_Extraction_and_Visualization_button_layout)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
              
        self.Feature_Extraction_and_Visualization_button.clicked.connect(self.show_screen1)

        self.Conventional_CAD_button = QPushButton(self)
        self.Conventional_CAD_button.setIcon(QIcon('imgs/Conventional_CAD_button.png'))
        self.Conventional_CAD_button.setMinimumSize(420,70)
        self.Conventional_CAD_button.setIconSize(QSize(45,45))
        self.Conventional_CAD_button.setText(" Conventional CAD")
        self.Conventional_CAD_button.setFont(font)
        self.Conventional_CAD_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.Conventional_CAD_button.clicked.connect(self.show_screen2)
        
        
        self.Conventional_CAD_button_layout = QHBoxLayout()
        self.Conventional_CAD_button_layout.setContentsMargins(10,0,0,0)
        self.Conventional_CAD_button_layout.addWidget(self.Conventional_CAD_button, alignment= Qt.AlignLeft)
        self.Conventional_CAD_button_layout_Widget = QWidget(self)
        self.Conventional_CAD_button_layout_Widget.setFixedWidth(420)
        self.Conventional_CAD_button_layout_Widget.setFixedHeight(70)
        self.Conventional_CAD_button_layout_Widget.setLayout(self.Conventional_CAD_button_layout)
        self.Conventional_CAD_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Conventional_CAD_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        font2 = QFont()
        font2.setPointSize(17)
        
        self.Conventional_RadioButtons_layout = QVBoxLayout()
        
        self.radio_buttons_Conventional = []
        self.radio_buttons_names_Conventional = ["Normal Vs OsteoArthritis", "Normal Vs Mild Vs Severe", "Five Classes","Fused Model"]

        for i in range(4):
            radio_button = QRadioButton(self.radio_buttons_names_Conventional[i])
            radio_button.setStyleSheet("color: darkgray;")
            radio_button.setFont(font2)
            radio_button.clicked.connect(lambda _, i=i: self.on_radio_button_Conventional_clicked(i))
            self.Conventional_RadioButtons_layout.addWidget(radio_button)
            self.radio_buttons_Conventional.append(radio_button)

            if i == 0:
                radio_button.setChecked(True)


        self.Ai_Automated_CAD_button = QPushButton(self)
        self.Ai_Automated_CAD_button.setIcon(QIcon('imgs/AI_Automated_CAD_button.png'))
        self.Ai_Automated_CAD_button.setMinimumSize(420,70)
        self.Ai_Automated_CAD_button.setIconSize(QSize(45,45))
        self.Ai_Automated_CAD_button.setText(" Ai Automated CAD")
        self.Ai_Automated_CAD_button.setFont(font)
        self.Ai_Automated_CAD_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.Ai_Automated_CAD_button.clicked.connect(self.show_screen3)
        
        self.Ai_Automated_CAD_button_layout = QHBoxLayout()
        self.Ai_Automated_CAD_button_layout.setContentsMargins(10,0,0,0)
        self.Ai_Automated_CAD_button_layout.addWidget(self.Ai_Automated_CAD_button, alignment= Qt.AlignLeft)
        self.Ai_Automated_CAD_button_layout_Widget = QWidget(self)
        self.Ai_Automated_CAD_button_layout_Widget.setFixedWidth(420)
        self.Ai_Automated_CAD_button_layout_Widget.setFixedHeight(70)
        self.Ai_Automated_CAD_button_layout_Widget.setLayout(self.Ai_Automated_CAD_button_layout)
        self.Ai_Automated_CAD_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Ai_Automated_CAD_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
              
        self.Reset_button = QPushButton(self)
        self.Reset_button.setIcon(QIcon('imgs/trash.svg'))
        self.Reset_button.setMinimumSize(420,70)
        self.Reset_button.setIconSize(QSize(45,45))
        self.Reset_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Reset_button.clicked.connect(self.Clear_All)
        self.Reset_button.setText(" Clear")
        self.Reset_button.setFont(font)
        self.Reset_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")

        self.Reset_button_layout = QHBoxLayout()
        self.Reset_button_layout.setContentsMargins(10,0,0,0)
        self.Reset_button_layout.addWidget(self.Reset_button, alignment= Qt.AlignLeft)
        self.Reset_button_layout_Widget = QWidget(self)
        self.Reset_button_layout_Widget.setFixedWidth(420)
        self.Reset_button_layout_Widget.setFixedHeight(70)
        self.Reset_button_layout_Widget.setLayout(self.Reset_button_layout)
        self.Reset_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Reset_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        
        
        # self.settings_button = QPushButton(self)
        # self.settings_button.setIcon(QIcon('imgs/settings.svg'))
        # self.settings_button.setMinimumSize(420,70)
        # self.settings_button.setIconSize(QSize(45,45))
        # self.settings_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        # self.settings_button.clicked.connect(self.settings)
        # self.settings_button.setText(" Settings")
        # self.settings_button.setFont(font)
        # self.settings_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")

        # self.settings_button_layout = QHBoxLayout()
        # self.settings_button_layout.setContentsMargins(10,0,0,0)
        # self.settings_button_layout.addWidget(self.settings_button, alignment= Qt.AlignLeft)
        # self.settings_button_layout_Widget = QWidget(self)
        # self.settings_button_layout_Widget.setFixedWidth(420)
        # self.settings_button_layout_Widget.setFixedHeight(70)
        # self.settings_button_layout_Widget.setLayout(self.settings_button_layout)
        # self.settings_button_layout_Widget.setStyleSheet("""
        #                                                                     background-color: transparent;
        #                                                                     border: none;
        #                                                                              """)
        # self.settings_button_layout_Widget.setStyleSheet("""
                                                                                     
                                                                                     
                                                                                     
        #                                                                              QWidget:hover {
        #                                                                                             background-color: rgba(64, 64, 64, 150);}
        #                                                                              """)
              
        
        
                   
        self.Windows_button_button44 = QPushButton(self)
        self.Windows_button_button44.setIcon(QIcon('imgs/eye-off.svg'))
        self.Windows_button_button44.setMinimumSize(420,70)
        self.Windows_button_button44.setIconSize(QSize(45,45))
        self.Windows_button_button44.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Windows_button_button44.setText(" Show Histograms and Visualizations")
        self.Windows_button_button44.setFont(font)
        self.Windows_button_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.Windows_button_button44.clicked.connect(self.Windows_button_Toggling_Visualizations)
        
        self.Windows_button_button44_layout = QHBoxLayout()
        self.Windows_button_button44_layout.setContentsMargins(10,0,0,0)
        self.Windows_button_button44_layout.addWidget(self.Windows_button_button44, alignment= Qt.AlignLeft)
        self.Windows_button_button44_layout_Widget = QWidget(self)
        self.Windows_button_button44_layout_Widget.setFixedWidth(420)
        self.Windows_button_button44_layout_Widget.setFixedHeight(70)
        self.Windows_button_button44_layout_Widget.setLayout(self.Windows_button_button44_layout)
        self.Windows_button_button44_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Windows_button_button44_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        
              
        
               
        self.Save_button44 = QPushButton(self)
        self.Save_button44.setIcon(QIcon('imgs/save.svg'))
        self.Save_button44.setMinimumSize(420,70)
        self.Save_button44.setIconSize(QSize(45,45))
        self.Save_button44.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Save_button44.setText(" Save All Figures")
        self.Save_button44.setFont(font)
        self.Save_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.Save_button44.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.Save_ALL)
        
        self.Save_button44_layout = QHBoxLayout()
        self.Save_button44_layout.setContentsMargins(10,0,0,0)
        self.Save_button44_layout.addWidget(self.Save_button44, alignment= Qt.AlignLeft)
        self.Save_button44_layout_Widget = QWidget(self)
        self.Save_button44_layout_Widget.setFixedWidth(420)
        self.Save_button44_layout_Widget.setFixedHeight(70)
        self.Save_button44_layout_Widget.setLayout(self.Save_button44_layout)
        self.Save_button44_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Save_button44_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        
              
              
              
              
        self.help_button = QPushButton(self)
        self.help_button.setIcon(QIcon('imgs/help.svg'))
        self.help_button.setMinimumSize(420,70)
        self.help_button.setIconSize(QSize(45,45))
        self.help_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.help_button.setText(" Help")
        self.help_button.setFont(font)
        self.help_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.help_button.clicked.connect(self.help)
        
        self.help_button_layout = QHBoxLayout()
        self.help_button_layout.setContentsMargins(10,0,0,0)
        self.help_button_layout.addWidget(self.help_button, alignment= Qt.AlignLeft)
        self.help_button_layout_Widget = QWidget(self)
        self.help_button_layout_Widget.setFixedWidth(420)
        self.help_button_layout_Widget.setFixedHeight(70)
        self.help_button_layout_Widget.setLayout(self.help_button_layout)
        self.help_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.help_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        
        self.exit_button = QPushButton(self)
        self.exit_button.setIcon(QIcon('imgs/log-out.svg'))
        self.exit_button.setMinimumSize(420,70)
        self.exit_button.setIconSize(QSize(45,45))
        self.exit_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.exit_button.clicked.connect(self.closeEvent)
        self.exit_button.setText(" Exit")
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
                     
        self.exit_button_layout = QHBoxLayout()
        self.exit_button_layout.setContentsMargins(10,0,0,0)
        self.exit_button_layout.addWidget(self.exit_button, alignment= Qt.AlignLeft)
        self.exit_button_layout_Widget = QWidget(self)
        self.exit_button_layout_Widget.setFixedWidth(420)
        self.exit_button_layout_Widget.setFixedHeight(70)
        self.exit_button_layout_Widget.setLayout(self.exit_button_layout)
        
        self.exit_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.exit_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)

        self.Ai_RadioButtons_layout = QVBoxLayout()
        self.radio_buttons_Ai = []
        self.radio_buttons_names_Ai = ["Normal Vs OsteoArthritis", "Normal Vs Mild Vs Severe", "Five Classes"]

        for i in range(3):
            radio_button = QRadioButton(self.radio_buttons_names_Ai[i])
            radio_button.setStyleSheet("color: darkgray;")
            radio_button.setFont(font2)
            radio_button.clicked.connect(lambda _, i=i: self.on_radio_button_AI_clicked(i))
            self.Ai_RadioButtons_layout.addWidget(radio_button)
            self.radio_buttons_Ai.append(radio_button)

            if i == 0:
                radio_button.setChecked(True)

        spacer_nav_reduced1 = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced3 = QSpacerItem(0, 500, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced4 = QSpacerItem(0, 230, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.spacer_nav_reduced3_layout = QVBoxLayout()
        self.spacer_nav_reduced3_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced3_layout.addSpacerItem(spacer_nav_reduced3)
        self.spacer_nav_reduced3_layout_Widget = QWidget(self)
        self.spacer_nav_reduced3_layout_Widget.setFixedWidth(70)
        self.spacer_nav_reduced3_layout_Widget.setFixedHeight(500)
        self.spacer_nav_reduced3_layout_Widget.setLayout(self.spacer_nav_reduced3_layout)
        self.spacer_nav_reduced3_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        

        self.spacer_nav_reduced4_layout = QVBoxLayout()
        self.spacer_nav_reduced4_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced4_layout.addSpacerItem(spacer_nav_reduced4)
        self.spacer_nav_reduced4_layout_Widget = QWidget(self)
        self.spacer_nav_reduced4_layout_Widget.setFixedWidth(70)
        self.spacer_nav_reduced4_layout_Widget.setFixedHeight(500)
        self.spacer_nav_reduced4_layout_Widget.setLayout(self.spacer_nav_reduced4_layout)
        self.spacer_nav_reduced4_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        
      
        self.Feature_Extraction_and_Visualization_button_reduced = QPushButton(self)
        self.Feature_Extraction_and_Visualization_button_reduced.setIcon(QIcon('imgs/Feature_Extraction_button.png'))
        self.Feature_Extraction_and_Visualization_button_reduced.setMinimumSize(45,45)
        self.Feature_Extraction_and_Visualization_button_reduced.setMaximumSize(45,45)
        self.Feature_Extraction_and_Visualization_button_reduced.setIconSize(QSize(45,45))
        self.Feature_Extraction_and_Visualization_button_reduced.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")         
        self.Feature_Extraction_and_Visualization_button_reduced.clicked.connect(self.show_screen1)

        self.Feature_Extraction_and_Visualization_button_reduced_layout = QHBoxLayout()
        self.Feature_Extraction_and_Visualization_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Feature_Extraction_and_Visualization_button_reduced_layout.addWidget(self.Feature_Extraction_and_Visualization_button_reduced, alignment= Qt.AlignCenter)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget = QWidget(self)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setFixedWidth(70)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setFixedHeight(70)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setLayout(self.Feature_Extraction_and_Visualization_button_reduced_layout)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     
                                                                                     
                                                                                     
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)

        self.Conventional_CAD_button_reduced = QPushButton(self)
        self.Conventional_CAD_button_reduced.setIcon(QIcon('imgs/Conventional_CAD_button.png'))
        self.Conventional_CAD_button_reduced.setMinimumSize(45,45)
        self.Conventional_CAD_button_reduced.setMaximumSize(45,45)
        self.Conventional_CAD_button_reduced.setIconSize(QSize(45,45))
        self.Conventional_CAD_button_reduced.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        self.Conventional_CAD_button_reduced.clicked.connect(self.show_screen2)
        
        self.Conventional_CAD_button_reduced_layout = QHBoxLayout()
        self.Conventional_CAD_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Conventional_CAD_button_reduced_layout.addWidget(self.Conventional_CAD_button_reduced, alignment= Qt.AlignCenter)
        self.Conventional_CAD_button_reduced_layout_Widget = QWidget(self)
        self.Conventional_CAD_button_reduced_layout_Widget.setFixedWidth(70)
        self.Conventional_CAD_button_reduced_layout_Widget.setFixedHeight(70)
        self.Conventional_CAD_button_reduced_layout_Widget.setLayout(self.Conventional_CAD_button_reduced_layout)
        self.Conventional_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Conventional_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
        self.Ai_Automated_CAD_button_reduced = QPushButton(self)
        self.Ai_Automated_CAD_button_reduced.setIcon(QIcon('imgs/Ai_Automated_CAD_button.png'))
        self.Ai_Automated_CAD_button_reduced.setMinimumSize(45,45)
        self.Ai_Automated_CAD_button_reduced.setMaximumSize(45,45)
        self.Ai_Automated_CAD_button_reduced.setIconSize(QSize(45,45))
        self.Ai_Automated_CAD_button_reduced.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        self.Ai_Automated_CAD_button_reduced.clicked.connect(self.show_screen3)

        self.Ai_Automated_CAD_button_reduced_layout = QHBoxLayout()
        self.Ai_Automated_CAD_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Ai_Automated_CAD_button_reduced_layout.addWidget(self.Ai_Automated_CAD_button_reduced, alignment= Qt.AlignCenter)
        self.Ai_Automated_CAD_button_reduced_layout_Widget = QWidget(self)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setFixedWidth(70)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setFixedHeight(70)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setLayout(self.Ai_Automated_CAD_button_reduced_layout)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
        
        
        
        
        
        
        
                
        self.Windows_button_button1 = QPushButton(self)
        self.Windows_button_button1.setIcon(QIcon('imgs/eye-off.svg'))
        self.Windows_button_button1.setMinimumSize(45,45)
        self.Windows_button_button1.setMaximumSize(45,45)
        self.Windows_button_button1.setIconSize(QSize(45,45))
        self.Windows_button_button1.clicked.connect(self.Windows_button_Toggling_Visualizations_nav_reduced)
        self.Windows_button_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        self.Windows_button_button1_layout = QHBoxLayout()
        self.Windows_button_button1_layout.setContentsMargins(10,0,0,0)
        self.Windows_button_button1_layout.addWidget(self.Windows_button_button1, alignment= Qt.AlignLeft)
        self.Windows_button_button1_layout_Widget = QWidget(self)
        self.Windows_button_button1_layout_Widget.setFixedWidth(70)
        self.Windows_button_button1_layout_Widget.setFixedHeight(70)
        self.Windows_button_button1_layout_Widget.setLayout(self.Windows_button_button1_layout)
        self.Windows_button_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Windows_button_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
              
              
              
              
              
              
              
                      
        self.Save_button_button1 = QPushButton(self)
        self.Save_button_button1.setIcon(QIcon('imgs/save.svg'))
        self.Save_button_button1.setMinimumSize(45,45)
        self.Save_button_button1.setMaximumSize(45,45)
        self.Save_button_button1.setIconSize(QSize(45,45))
        self.Save_button_button1.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.Save_ALL)
        self.Save_button_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        self.Save_button_button1_layout = QHBoxLayout()
        self.Save_button_button1_layout.setContentsMargins(10,0,0,0)
        self.Save_button_button1_layout.addWidget(self.Save_button_button1, alignment= Qt.AlignLeft)
        self.Save_button_button1_layout_Widget = QWidget(self)
        self.Save_button_button1_layout_Widget.setFixedWidth(70)
        self.Save_button_button1_layout_Widget.setFixedHeight(70)
        self.Save_button_button1_layout_Widget.setLayout(self.Save_button_button1_layout)
        self.Save_button_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Save_button_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
              
              
              
              
              
              
              
              
              
        
        self.Reset_button1 = QPushButton(self)
        self.Reset_button1.setIcon(QIcon('imgs/trash.svg'))
        self.Reset_button1.setMinimumSize(45,45)
        self.Reset_button1.setMaximumSize(45,45)
        self.Reset_button1.setIconSize(QSize(45,45))
        self.Reset_button1.clicked.connect(self.Clear_All)
        self.Reset_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        self.Reset_button1_layout = QHBoxLayout()
        self.Reset_button1_layout.setContentsMargins(10,0,0,0)
        self.Reset_button1_layout.addWidget(self.Reset_button1, alignment= Qt.AlignLeft)
        self.Reset_button1_layout_Widget = QWidget(self)
        self.Reset_button1_layout_Widget.setFixedWidth(70)
        self.Reset_button1_layout_Widget.setFixedHeight(70)
        self.Reset_button1_layout_Widget.setLayout(self.Reset_button1_layout)
        self.Reset_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Reset_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
              
              
              
              
              
        # self.settings_button1 = QPushButton(self)
        # self.settings_button1.setIcon(QIcon('imgs/settings.svg'))
        # self.settings_button1.setMinimumSize(45,45)
        # self.settings_button1.setMaximumSize(45,45)
        # self.settings_button1.setIconSize(QSize(45,45))
        # self.settings_button1.clicked.connect(self.settings)
        # self.settings_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        # self.settings_button1_layout = QHBoxLayout()
        # self.settings_button1_layout.setContentsMargins(10,0,0,0)
        # self.settings_button1_layout.addWidget(self.settings_button1, alignment= Qt.AlignLeft)
        # self.settings_button1_layout_Widget = QWidget(self)
        # self.settings_button1_layout_Widget.setFixedWidth(70)
        # self.settings_button1_layout_Widget.setFixedHeight(70)
        # self.settings_button1_layout_Widget.setLayout(self.settings_button1_layout)
        # self.settings_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        # self.settings_button1_layout_Widget.setStyleSheet("""
        #                                                                              QWidget:hover {
        #                                                                                             background-color: rgba(64, 64, 64, 150);}
        #                                                                              """)
              
        self.help_button1 = QPushButton(self)
        self.help_button1.setIcon(QIcon('imgs/help.svg'))
        self.help_button1.setMinimumSize(45,45)
        self.help_button1.setMaximumSize(45,45)
        self.help_button1.setIconSize(QSize(45,45))
        self.help_button1.clicked.connect(self.help)
        self.help_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        self.help_button1_layout = QHBoxLayout()
        self.help_button1_layout.setContentsMargins(10,0,0,0)
        self.help_button1_layout.addWidget(self.help_button1, alignment= Qt.AlignLeft)
        self.help_button1_layout_Widget = QWidget(self)
        self.help_button1_layout_Widget.setFixedWidth(70)
        self.help_button1_layout_Widget.setFixedHeight(70)
        self.help_button1_layout_Widget.setLayout(self.help_button1_layout)
        self.help_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.help_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)
        
        self.exit_button1 = QPushButton(self)
        self.exit_button1.setIcon(QIcon('imgs/log-out.svg'))
        self.exit_button1.setMinimumSize(45,45)
        self.exit_button1.setMaximumSize(45,45)
        self.exit_button1.setIconSize(QSize(45,45))
        self.exit_button1.clicked.connect(self.closeEvent)
        self.exit_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 200); border: none; background: transparent; border-radius: 50px; }")
        
        self.exit_button1_layout = QHBoxLayout()
        self.exit_button1_layout.setContentsMargins(10,0,0,0)
        self.exit_button1_layout.addWidget(self.exit_button1, alignment= Qt.AlignLeft)
        self.exit_button1_layout_Widget = QWidget(self)
        self.exit_button1_layout_Widget.setFixedWidth(70)
        self.exit_button1_layout_Widget.setFixedHeight(70)
        self.exit_button1_layout_Widget.setLayout(self.exit_button1_layout)
        self.exit_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.exit_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 64, 64, 150);}
                                                                                     """)

        self.nav_layout_reduced.addSpacerItem(spacer_nav_reduced1)
        self.nav_layout_reduced.addWidget(self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.Conventional_CAD_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced3_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Ai_Automated_CAD_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced4_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Windows_button_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Save_button_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Reset_button1_layout_Widget)
        # self.nav_layout_reduced.addWidget(self.settings_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.help_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.exit_button1_layout_Widget)
        
        self.Conventional_RadioButtons_layout_widget = QWidget(self)
        self.Conventional_RadioButtons_layout_widget.setFixedHeight(200)
        self.Conventional_RadioButtons_layout_widget.setLayout(self.Conventional_RadioButtons_layout)
        
        self.Ai_RadioButtons_layout_widget = QWidget(self)
        self.Ai_RadioButtons_layout_widget.setFixedHeight(200)
        self.Ai_RadioButtons_layout_widget.setLayout(self.Ai_RadioButtons_layout)

        spacer_nav1 = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav3_H = QSpacerItem(50, 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer_nav4_H = QSpacerItem(50, 0, QSizePolicy.Fixed, QSizePolicy.Fixed)


        self.Equalize_button = QPushButton(self)
        self.Equalize_button.setIcon(QIcon('imgs/check-square.svg'))
        self.Equalize_button.setMinimumSize(35,35)
        self.Equalize_button.setMaximumSize(35,35)
        self.Equalize_button.setIconSize(QSize(35,35))
        self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Equalize_button.clicked.connect(self.toggle_intensity_normalization_Equalize)
        
        self.Equalization_label = QLabel("Apply intensity Equalization", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        self.Equalization_label.setFont(font)
        self.Equalization_label.setMinimumSize(300,35)
        self.Equalization_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        
        self.Auto_F_Extractor_Single = QPushButton(self)
        self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/enable-mode.png'))
        self.Auto_F_Extractor_Single.setMinimumSize(40,40)
        self.Auto_F_Extractor_Single.setMaximumSize(40,40)
        self.Auto_F_Extractor_Single.setIconSize(QSize(40,40))
        self.Auto_F_Extractor_Single.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Auto_F_Extractor_Single.clicked.connect(self.switch_mode_Auto_Mode)
        
        
        self.Auto_F_Extractor_Dataset = QPushButton(self)
        self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/disable-mode.svg'))
        self.Auto_F_Extractor_Dataset.setMinimumSize(40,40)
        self.Auto_F_Extractor_Dataset.setMaximumSize(40,40)
        self.Auto_F_Extractor_Dataset.setIconSize(QSize(40,40))
        self.Auto_F_Extractor_Dataset.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Auto_F_Extractor_Dataset.clicked.connect(self.switch_mode_Auto_Mode)
        
        self.Single_img_label = QLabel("Single image Mode", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        self.Single_img_label.setFont(font)
        self.Single_img_label.setMinimumSize(300,35)
        self.Single_img_label.setStyleSheet("QLabel { text-align:right; color:darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.Single_img_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)


        self.Auto_mode_label = QLabel("Dataset automatic Mode", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        self.Auto_mode_label.setFont(font)
        self.Auto_mode_label.setMinimumSize(300,35)
        self.Auto_mode_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.Auto_mode_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.nav_1st_Horizontal_Layout = QVBoxLayout()
        self.nav_1st_Horizontal_Layout.setContentsMargins(35, 35, 35, 35)
        self.nav_1st_Horizontal_Layout.setAlignment(Qt.AlignCenter)
        
        self.nav_1st_Horizontal_Widget = QWidget()
        self.nav_1st_Horizontal_Widget.setLayout(self.nav_1st_Horizontal_Layout)
        self.nav_1st_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.nav_1st_Horizontal_Widget_Spacer = QSpacerItem(0, 200, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.nav_1st_Horizontal_Layout_4 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_4.setContentsMargins(0,0,0,0)
        self.nav_1st_Horizontal_Widget_4 = QWidget()
        self.nav_1st_Horizontal_Widget_4.setLayout(self.nav_1st_Horizontal_Layout_4)
        self.nav_1st_Horizontal_Widget_4.setFixedHeight(40)
        self.nav_1st_Horizontal_Widget_4.setStyleSheet("background-color: transparent; border: none;")
        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_4)
        self.nav_1st_Horizontal_Layout_4.addWidget(self.Equalization_label)
        self.nav_1st_Horizontal_Layout_4.addWidget(self.Equalize_button)

        self.nav_1st_Horizontal_Layout_1 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_1.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_1 = QWidget()
        self.nav_1st_Horizontal_Widget_1.setLayout(self.nav_1st_Horizontal_Layout_1)
        self.nav_1st_Horizontal_Widget_1.setFixedHeight(40)
        self.nav_1st_Horizontal_Widget_1.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_1)
        self.nav_1st_Horizontal_Layout_1.addWidget(self.Single_img_label)
        self.nav_1st_Horizontal_Layout_1.addWidget(self.Auto_F_Extractor_Single)
        
        self.nav_1st_Horizontal_Layout_2 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_2.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_2 = QWidget()
        self.nav_1st_Horizontal_Widget_2.setLayout(self.nav_1st_Horizontal_Layout_2)
        self.nav_1st_Horizontal_Widget_2.setFixedHeight(40)
        self.nav_1st_Horizontal_Widget_2.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_2)
        
        self.nav_1st_Horizontal_Layout_2.addWidget(self.Auto_mode_label)        
        self.nav_1st_Horizontal_Layout_2.addWidget(self.Auto_F_Extractor_Dataset)

        nav_2nd_Horizontal_Layout = QHBoxLayout()
        self.nav_2nd_Horizontal_Widget = QWidget()
        self.nav_2nd_Horizontal_Widget.setLayout(nav_2nd_Horizontal_Layout)
        self.nav_2nd_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_2nd_Horizontal_Layout.addSpacerItem(spacer_nav3_H)
        nav_2nd_Horizontal_Layout.addWidget(self.Conventional_RadioButtons_layout_widget)

        nav_3rd_Horizontal_Layout = QHBoxLayout()
        self.nav_3rd_Horizontal_Widget = QWidget()
        self.nav_3rd_Horizontal_Widget.setLayout(nav_3rd_Horizontal_Layout)
        self.nav_3rd_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_3rd_Horizontal_Layout.addSpacerItem(spacer_nav4_H)
        nav_3rd_Horizontal_Layout.addWidget(self.Ai_RadioButtons_layout_widget)
        
        self.nav_layout.addSpacerItem(spacer_nav1)
        self.nav_layout.addWidget(self.Feature_Extraction_and_Visualization_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_1st_Horizontal_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.Conventional_CAD_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_2nd_Horizontal_Widget, stretch=1)
        self.nav_layout.addWidget(self.Ai_Automated_CAD_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_3rd_Horizontal_Widget,  stretch=1)
        self.nav_layout.addWidget(self.Windows_button_button44_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.Save_button44_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.Reset_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.help_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.exit_button_layout_Widget, alignment= Qt.AlignLeft)
        
        self.nav_Widget_reduced = QWidget(self)
        self.nav_Widget = QWidget(self)
        self.nav_Widget.hide()
        self.nav_Widget_reduced.setFixedWidth(65)

        self.menu_button = QPushButton(self)
        self.menu_button.setIcon(QIcon("imgs/menu.svg"))
        self.menu_button.setFixedSize(35, 35)
        self.menu_button.setIconSize(self.menu_button.size())
        self.menu_button.setStyleSheet("border: none;background-color: transparent;border-radius: 50px;")
        self.menu_button.clicked.connect(self.Toggle_nav_Widget)

        self.label_Title_name = QLabel("Knee OsteoArthritis Diagnosis")
        self.label_Title_name.setStyleSheet("color: rgba(255, 255, 255, 150);")
        font = QFont()
        font.setBold(True)
        font.setPointSize(15)
        self.label_Title_name.setFont(font)

        self.Minimize_button = QPushButton(self)
        self.Minimize_button.setIcon(QIcon('imgs/minimize.svg'))  # Update the icon path
        self.Minimize_button.setMinimumSize(25, 25)
        self.Minimize_button.setMaximumSize(25, 25)
        self.Minimize_button.setIconSize(QSize(25, 25))
        self.Minimize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Minimize_button.clicked.connect(self.toggleMinimized)  # Connect to the new toggleMinimized method
        
        self.Expand_button = QPushButton(self)
        self.Expand_button.setIcon(QIcon('imgs/reduce.svg'))
        self.Expand_button.setMinimumSize(25,25)
        self.Expand_button.setMaximumSize(25,25)
        self.Expand_button.setIconSize(QSize(25,25))
        self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Expand_button.clicked.connect(self.Expand_Function)
        
        self.Restart_button = QPushButton(self)
        self.Restart_button.setIcon(QIcon('imgs/restart.svg'))
        self.Restart_button.setMinimumSize(25,25)
        self.Restart_button.setMaximumSize(25,25)
        self.Restart_button.setIconSize(QSize(25,25))
        self.Restart_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Restart_button.clicked.connect(self.restart_code)
        
        self.EXIT_button = QPushButton(self)
        self.EXIT_button.setIcon(QIcon('imgs/x.svg'))
        self.EXIT_button.setMinimumSize(25,25)
        self.EXIT_button.setMaximumSize(25,25)
        self.EXIT_button.setIconSize(QSize(25,25))
        self.EXIT_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.EXIT_button.clicked.connect(self.closeEvent)
        
        self.titleBar_layout = QHBoxLayout()
        spacer_titleBar = QSpacerItem(800, 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer_titleBar2 = QSpacerItem(50, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        self.titleBar_layout.addWidget(self.menu_button)
        self.titleBar_layout.addItem(spacer_titleBar)
        self.titleBar_layout.addWidget(self.label_Title_name)
        self.titleBar_layout.addItem(spacer_titleBar2)
        self.titleBar_layout.addWidget(self.Restart_button)
        self.titleBar_layout.addWidget(self.Minimize_button)
        self.titleBar_layout.addWidget(self.Expand_button)
        self.titleBar_layout.addWidget(self.EXIT_button)
        
        self.title_bar = QWidget(self)
        self.title_bar.setLayout(self.titleBar_layout)
        self.title_bar.setFixedHeight(80)
        
        self.central_layout.addWidget(self.title_bar)

        gradient_style2 = """
            background: qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 75), stop:1 rgba(0, 0, 0, 0));
        """
                
        self.nav_Widget_reduced.setLayout(self.nav_layout_reduced)
        self.nav_Widget.setLayout(self.nav_layout)
        self.nav_Widget_reduced.setStyleSheet(gradient_style2)
        self.nav_Widget.setStyleSheet(gradient_style2)

        self.Main_layout = QHBoxLayout()
        self.screen_layout = QVBoxLayout()
        
        self.Main_layout.addWidget(self.nav_Widget_reduced)
        self.Main_layout.addWidget(self.nav_Widget)
        self.Main_layout.addLayout(self.screen_layout)
        self.central_layout.addLayout(self.Main_layout)

        self.drag_position = None
        self.setMouseTracking(True)

        self.current_screen = None
        self.setCentralWidget(self.central_widget)
        self.show_screen1()
        
        
    def Windows_button_Toggling_Visualizations(self):
        if (self.switch2 == False):
            self.Windows_button_button44.setIcon(QIcon('imgs/eye.svg'))
            self.Windows_button_button44.setMinimumSize(420,70)
            self.Windows_button_button44.setIconSize(QSize(45,45))
            self.Windows_button_button44.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            self.Windows_button_button44.setText(" Hide Histograms and Visualizations")
            self.Windows_button_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = True
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
        else:
            self.Windows_button_button44.setIcon(QIcon('imgs/eye-off.svg'))
            self.Windows_button_button44.setMinimumSize(420,70)
            self.Windows_button_button44.setIconSize(QSize(45,45))
            self.Windows_button_button44.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            self.Windows_button_button44.setText(" Show Histograms and Visualizations")
            self.Windows_button_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = False
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
            
    def Windows_button_Toggling_Visualizations_nav_reduced(self):
        if (self.switch2 == False):
            self.Windows_button_button1.setIcon(QIcon('imgs/eye.svg'))
            self.Windows_button_button1.setMinimumSize(45,45)
            self.Windows_button_button1.setMaximumSize(45,45)
            self.Windows_button_button1.setIconSize(QSize(45,45))
            self.Windows_button_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = True
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
        else:
            self.Windows_button_button1.setIcon(QIcon('imgs/eye-off.svg'))
            self.Windows_button_button1.setMinimumSize(45,45)
            self.Windows_button_button1.setMaximumSize(45,45)
            self.Windows_button_button1.setIconSize(QSize(45,45))
            self.Windows_button_button1.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = False
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
                
    def switch_mode_Auto_Mode(self):
        if (self.switch == True):
            self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/disable-mode.svg'))
            self.Auto_F_Extractor_Single.setMinimumSize(40,40)
            self.Auto_F_Extractor_Single.setMaximumSize(40,40)
            self.Auto_F_Extractor_Single.setIconSize(QSize(40,40))
            self.Auto_F_Extractor_Single.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
            self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/enable-mode.png'))
            self.Auto_F_Extractor_Dataset.setMinimumSize(40,40)
            self.Auto_F_Extractor_Dataset.setMaximumSize(40,40)
            self.Auto_F_Extractor_Dataset.setIconSize(QSize(40,40))
            self.Auto_F_Extractor_Dataset.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
            text = "It will be minimized, then restored again after the automation completes.\n"
            msgBox = CustomMessageBox2()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Automatic Mode")
            msgBox.setWindowIcon(QIcon("imgs/alert-triangle.png"))
            msgBox.setText(text)
            font = QFont()
            font.setPointSize(11)
            msgBox.setFont(font)
            msgBox.addButton(QMessageBox.Ok)
            msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            msgBox.exec_()
            self.switch= False
            self.variableChanged_Auto_mode.emit(self.switch)
            
        else:
            self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/enable-mode.png'))
            self.Auto_F_Extractor_Single.setMinimumSize(40,40)
            self.Auto_F_Extractor_Single.setMaximumSize(40,40)
            self.Auto_F_Extractor_Single.setIconSize(QSize(40,40))
            self.Auto_F_Extractor_Single.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            self.Auto_F_Extractor_Single.setToolTip("switch to Data set Automatic Feature Extractor")
            
            self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/disable-mode.svg'))
            self.Auto_F_Extractor_Dataset.setMinimumSize(40,40)
            self.Auto_F_Extractor_Dataset.setMaximumSize(40,40)
            self.Auto_F_Extractor_Dataset.setIconSize(QSize(40,40))
            self.Auto_F_Extractor_Dataset.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
            self.switch = True
            self.variableChanged_Auto_mode.emit(self.switch)
            
    def toggle_intensity_normalization_Equalize(self):
        if self.perform_intensity_normalization == False:
            self.perform_intensity_normalization = True
            self.variableChanged_Equalize_Feature_Visualization.emit(self.perform_intensity_normalization)
            
            self.Equalize_button.setIcon(QIcon('imgs/check-square.svg'))
            self.Equalize_button.setMinimumSize(35,35)
            self.Equalize_button.setMaximumSize(35,35)
            self.Equalize_button.setIconSize(QSize(35,35))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        else:

            self.perform_intensity_normalization = False
            self.variableChanged_Equalize_Feature_Visualization.emit(self.perform_intensity_normalization)

            self.Equalize_button.setIcon(QIcon('imgs/square.svg'))
            self.Equalize_button.setMinimumSize(35,35)
            self.Equalize_button.setMaximumSize(35,35)
            self.Equalize_button.setIconSize(QSize(35,35))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

            self.Feature_Extraction_and_Visualization_Screen.Intensity_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.Binarization_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.LBPandLTPImgsWindow.LBPImgWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.LBPandLTPImgsWindow.LTPImgWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.LBPHistogramWindow.LBPHistogramWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.LTPHistogramWindow.LTPHistogramWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.HistogramOfOrientedGradientsImage.HOG_image_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.HOGHistogram.HOGHistogram_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.Binarization_label.setText("Binarization")
            self.Feature_Extraction_and_Visualization_Screen.Intensity_label.setText("Equalization")
            self.Feature_Extraction_and_Visualization_Screen.LBPandLTPImgsWindow.LBPImgWindow_label.setText("LBP")
            self.Feature_Extraction_and_Visualization_Screen.LBPandLTPImgsWindow.LTPImgWindow_label.setText("LTP")
            self.Feature_Extraction_and_Visualization_Screen.LBPHistogramWindow.LBPHistogramWindow_label.setText("LBP Histogram Window")
            self.Feature_Extraction_and_Visualization_Screen.LTPHistogramWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
            self.Feature_Extraction_and_Visualization_Screen.HistogramOfOrientedGradientsImage.HOG_image_label.setText("HOG Image")
            self.Feature_Extraction_and_Visualization_Screen.HOGHistogram.HOGHistogram_label.setText("HOG Histogram")
       
        if self.Feature_Extraction_and_Visualization_Screen.image_path is not None:
            self.Feature_Extraction_and_Visualization_Screen.load_image(self.Feature_Extraction_and_Visualization_Screen.image_path)
        else:
            pass
        

    def show_screen1(self):
        self.clear_screen_layout()
        
        self.nav_1st_Horizontal_Widget.setVisible(True)
        self.nav_2nd_Horizontal_Widget.setVisible(False)
        self.nav_3rd_Horizontal_Widget.setVisible(False)
        self.Ai_RadioButtons_layout_widget.setVisible(False)
        
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        
        self.current_screen = self.Feature_Extraction_and_Visualization_Screen

        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def show_screen2(self):
        self.clear_screen_layout()
        
        self.nav_1st_Horizontal_Widget.setVisible(False)
        self.nav_2nd_Horizontal_Widget.setVisible(True)
        self.nav_3rd_Horizontal_Widget.setVisible(False)
        
        self.spacer_nav_reduced3_layout_Widget.setVisible(True)
        self.spacer_nav_reduced4_layout_Widget.setVisible(False)

        self.current_screen = self.Conventional_CAD_Screen
        
        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def show_screen3(self):
        self.clear_screen_layout()
        
        self.nav_1st_Horizontal_Widget.setVisible(False)
        self.nav_2nd_Horizontal_Widget.setVisible(False)
        self.nav_3rd_Horizontal_Widget.setVisible(True)
        self.Ai_RadioButtons_layout_widget.setVisible(True)
        
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        
        self.current_screen = self.AI_Automated_CAD_Screen
        
        self.AI_Automated_CAD_Screen.on_button_click()
        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def clear_screen_layout(self):
        while self.screen_layout.count() > 0:
            item = self.screen_layout.takeAt(0)
            widget = item.widget()
            if widget:
                self.screen_layout.removeWidget(widget)
                widget.setParent(None)

    def Toggle_nav_Widget(self):
        self.nav_Widget_reduced.setVisible(not self.nav_Widget_reduced.isVisible())
        self.nav_Widget.setVisible(not self.nav_Widget.isVisible())
        
    def on_radio_button_Conventional_clicked(self, option):
        self.FirstSetVariable = option + 1
        self.variableChanged.emit(self.FirstSetVariable)

    def on_radio_button_AI_clicked(self, option):
        self.SecondSetVariable = option + 1
        self.variableChanged2.emit(self.SecondSetVariable)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.rect().contains(event.pos()):
            self.Expand_Function()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isFullScreen:
            self.drag_position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

    def toggleMinimized(self):
        if self.isMinimized():
            self.showNormal()
        else:
            self.showMinimized()
# __________________________________________ resize to Full Screen _______________________________________________
    def resizeEvent(self, event):
        pixmap1 = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(pixmap1)
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())
                    
    def restart_code(self):
        msgBox = CustomMessageBox3()
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            QApplication.quit()
            subprocess.Popen([sys.executable] + sys.argv)
        else:
            pass
        
    # def Settings(self):
        # if not hasattr(self, 'SettingsWindow') or not self.SettingsWindow.isVisible():
        #     self.SettingsWindow = SettingsWindow()
        # self.SettingsWindow.setVisible(not self.SettingsWindow.isVisible())
            
    def Clear_All(self):
        self.Feature_Extraction_and_Visualization_Screen.image_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.JSN_label.disableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.YOLO_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.Intensity_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.Binarization_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.edge_label.enableEffects() 
        
        self.Feature_Extraction_and_Visualization_Screen.image_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.JSN_label.disableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.YOLO_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.Intensity_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.Binarization_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.edge_label.leaveEvent(True) 
        
        self.Feature_Extraction_and_Visualization_Screen.on_button_click()
        self.AI_Automated_CAD_Screen.on_button_click()
        self.Conventional_CAD_Screen.on_button_click()
        
    def help(self):
        self.HelpWindow.setVisible(not self.HelpWindow.isVisible())
        
    def closeEvent(self, event):
        msgBox = CustomMessageBox()
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            exit(0)
        else:
            pass
        
    def Expand_Function(self):
        if self.isFullScreen:
            self.setWindowFlag(Qt.FramelessWindowHint)
            self.showNormal()
            self.isFullScreen = False
            self.Expand_button.setIcon(QIcon('imgs/expand.svg'))
            self.Expand_button.setMinimumSize(20,20)
            self.Expand_button.setMaximumSize(20,20)
            self.Expand_button.setIconSize(QSize(20,20))
            self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            self.isFullScreen = True
            self.Expand_button.setIcon(QIcon('imgs/reduce.svg'))
            self.Expand_button.setMinimumSize(25,25)
            self.Expand_button.setMaximumSize(25,25)
            self.Expand_button.setIconSize(QSize(25,25))
            self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen:
            self.Expand_Function()
        else:
            super().keyPressEvent(event)
#  __________________________________________________ main() func. ____________________________________________________

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = HomeScreen()
    main_window.show()
    sys.exit(app.exec_())