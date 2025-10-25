import sys, time, random
import numpy as np

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QPainter, QPen, QPixmap, QColor
from PyQt6.QtCore import Qt, QPoint, QTimer

from model_io import save_model, load_model
from train import train, classify_image


class AnimatedBackground(QWidget):
    def __init__(self, parent=None, num_waves=5):
        super().__init__(parent)
        self.num_waves = num_waves
        self.amplitudes = [random.uniform(10, 30) for _ in range(num_waves)]
        self.frequencies = [random.uniform(0.01, 0.03) for _ in range(num_waves)]
        self.phases = [0 for _ in range(num_waves)]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)

    def update_animation(self):
        for i in range(self.num_waves):
            self.phases[i] += 0.05
            self.amplitudes[i] += random.uniform(-0.5, 0.5)
            self.amplitudes[i] = max(5, min(50, self.amplitudes[i]))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(11, 11, 11))  # dark background

        pen = QPen(QColor(255, 255, 255, 120))
        pen.setWidth(2)
        painter.setPen(pen)

        width, height = self.width(), self.height()

        for i in range(self.num_waves):
            path_y = []
            for x in range(width):
                y = height / 2 + self.amplitudes[i] * np.sin(2 * np.pi * self.frequencies[i] * x + self.phases[i])
                path_y.append(y)
            for x in range(1, width):
                painter.drawLine(x-1, int(path_y[x-1]), x, int(path_y[x]))


class DrawingPanel(QWidget):
    def __init__(self, width=500, height=500):
        super().__init__()

        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.GlobalColor.black)
        self.last_point = QPoint()
        self.pen = QPen(Qt.GlobalColor.white, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()
    

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.pixmap)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            painter.end()
            self.update()
    

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
    
    
    def clear(self):
        self.pixmap.fill(Qt.GlobalColor.black)
        self.update()


    def get_numpy_image(self):
        qimage = self.pixmap.toImage()
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        img = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img_gray = np.mean(img[:, :, :3], axis=2) / 255.0

        return img_gray


class QuantumClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pictura Quantica: Quantum Image Classifier")
        self.setStyleSheet("background-color: #111111; color: #FFFFFF; font-size: 14px;")
        self.setGeometry(100, 100, 600, 500)

        self.background = AnimatedBackground(self)
        self.background.setGeometry(0, 0, 600, 500)

        self.model = None
        self.categories = None
        self.pca = None
        self.scaler_std = None
        self.scaler_quantum = None

        # Set up the components.

        self.panel = DrawingPanel()
        self.result_label = QLabel("Train/load a model before classifying.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; padding: 8px;")

        # Set up the buttons.

        self.classify_button = QPushButton("Classify")
        self.clear_button = QPushButton("Clear")
        self.train_button = QPushButton("Train Model")
        self.save_button = QPushButton("Save Model")
        self.load_button = QPushButton("Load Model")

        for btn in [self.classify_button, self.clear_button, self.train_button, self.save_button, self.load_button]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    color: white;
                    border: 1px solid #555;
                    border-radius: 6px;
                    padding: 6px;
                }
                QPushButton:hover { background-color: #444; }
            """)

        # Set up the layouts.

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.classify_button)
        btn_layout.addWidget(self.clear_button)
        btn_layout.addWidget(self.train_button)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.load_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.panel, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)

        # Connect the buttons.

        self.classify_button.clicked.connect(self.classify)
        self.clear_button.clicked.connect(self.panel.clear)
        self.train_button.clicked.connect(self.train_model)
        self.save_button.clicked.connect(self.save_current_model)
        self.load_button.clicked.connect(self.load_existing_model)


    def classify(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded or trained.")
            return

        image = self.panel.get_numpy_image()
        classification = classify_image(image, self.model, self.pca, self.scaler_std, self.scaler_quantum)
        
        self.result_label.setText(f"Predicted: {classification}")


    def train_model(self):
        try:
            start_time = time.time()
            self.model, self.categories, self.pca, self.scaler_std, self.scaler_quantum = train()
            end_time = time.time()
            time_elapsed = end_time - start_time

            QMessageBox.information(self, "Info", f"Training complete. (Elapsed Time: {time_elapsed:.3f} seconds)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed:\n{e}")


    def save_current_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model to save.")
            return
        
        save_model(self.model, self.categories, self.pca, self.scaler_std, self.scaler_quantum)
        QMessageBox.information(self, "Info", "Model saved successfully.")


    def load_existing_model(self):
        try:
            self.model, self.categories, self.pca, self.scaler_std, self.scaler_quantum = load_model()
            QMessageBox.information(self, "Info", "Model loaded successfully.")
        except RuntimeError as e:
            QMessageBox.warning(self, "Warning", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuantumClassifierGUI()
    window.show()

    sys.exit(app.exec())
