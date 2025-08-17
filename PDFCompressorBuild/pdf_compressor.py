import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QProgressBar, QComboBox, QSpinBox, QMessageBox,
                           QLineEdit, QGroupBox, QRadioButton, QButtonGroup,
                           QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import tempfile
import shutil
import logging
import numpy as np

class PDFCompressorWorker(QThread):
    """Worker thread to handle PDF compression without freezing the UI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, output_path, compression_settings):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.compression_settings = compression_settings
        
    def optimize_scanned_image(self, image):
        """Apply additional optimizations for scanned documents"""
        try:
            if self.compression_settings.get('optimize_scans', False):
                # Convert to grayscale if specified (before other optimizations)
                if self.compression_settings.get('grayscale', False):
                    image = image.convert('L')
                
                # Get original DPI or default to 300
                original_dpi = image.info.get('dpi', (300, 300))[0]
                if original_dpi == 0:
                    original_dpi = 300
                
                # Aggressive downsampling for very high DPI
                if original_dpi > 200:
                    scale_factor = 200 / original_dpi
                    new_size = (int(image.width * scale_factor), 
                              int(image.height * scale_factor))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                if image.mode == 'L':  # Grayscale optimization
                    # Convert to numpy array for advanced processing
                    img_array = np.array(image)
                    
                    # Normalize contrast
                    p2, p98 = np.percentile(img_array, (2, 98))
                    img_array = np.clip(((img_array - p2) / (p98 - p2) * 255), 0, 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
                    
                    # Enhance edges for text
                    image = image.filter(ImageFilter.EDGE_ENHANCE)
                    
                    # Reduce noise while preserving edges
                    image = image.filter(ImageFilter.MedianFilter(size=3))
                    
                    # Binarize text-heavy regions while preserving grayscale in image regions
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                    
                    # Reduce to optimal number of gray levels
                    image = image.quantize(colors=8, method=2).convert('L')
                    
                else:  # Color optimization
                    # Enhance image
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.2)
                    
                    # Color quantization
                    image = image.quantize(colors=32, method=2).convert('RGB')
                    
                    # Selective blur to reduce noise while preserving edges
                    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                
                # Final optimization pass
                image = ImageOps.autocontrast(image, cutoff=0.5)
                
            elif self.compression_settings.get('grayscale', False):
                # Simple grayscale conversion if no optimization
                image = image.convert('L')
            
            return image
            
        except Exception as e:
            logging.error(f"Image optimization error: {str(e)}")
            return image
            
    def run(self):
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_path(self.input_path)
                total_images = len(images)
                
                compressed_images = []
                for i, image in enumerate(images):
                    # Apply optimizations
                    compressed_image = self.optimize_scanned_image(image)
                    
                    # Save compressed image with optimal settings
                    temp_path = os.path.join(temp_dir, f'page_{i}.jpg')
                    
                    if self.compression_settings['mode'] == 'size':
                        target_size = self.compression_settings['value'] * 1024 * 1024  # Convert MB to bytes
                        target_size_per_page = target_size / total_images
                        
                        # Initialize quality
                        quality = 60  # Start with moderate quality for size mode
                        
                        # Save with initial quality
                        compressed_image.save(temp_path, 'JPEG',
                                           quality=quality,
                                           optimize=True,
                                           progressive=True,
                                           subsampling='4:2:0')
                        
                        current_size = os.path.getsize(temp_path)
                        
                        # Binary search for appropriate quality if needed
                        if current_size != target_size_per_page:
                            min_quality = 1
                            max_quality = 60  # Cap maximum quality
                            best_quality = quality
                            best_size_diff = abs(current_size - target_size_per_page)
                            
                            while min_quality <= max_quality:
                                current_quality = (min_quality + max_quality) // 2
                                
                                compressed_image.save(temp_path, 'JPEG',
                                                   quality=current_quality,
                                                   optimize=True,
                                                   progressive=True,
                                                   subsampling='4:2:0')
                                
                                current_size = os.path.getsize(temp_path)
                                size_diff = abs(current_size - target_size_per_page)
                                
                                if size_diff < best_size_diff:
                                    best_size_diff = size_diff
                                    best_quality = current_quality
                                
                                if current_size > target_size_per_page:
                                    max_quality = current_quality - 1
                                else:
                                    min_quality = current_quality + 1
                            
                            # Use the best quality found
                            compressed_image.save(temp_path, 'JPEG',
                                               quality=best_quality,
                                               optimize=True,
                                               progressive=True,
                                               subsampling='4:2:0')
                    
                    elif self.compression_settings['mode'] == 'percentage':
                        quality = self.compression_settings['value']
                        compressed_image.save(temp_path, 'JPEG',
                                           quality=quality,
                                           optimize=True,
                                           progressive=True,
                                           subsampling='4:2:0')
                    
                    else:  # resolution mode
                        quality = 60  # Use moderate quality for resolution mode
                        compressed_image.save(temp_path, 'JPEG',
                                           quality=quality,
                                           optimize=True,
                                           progressive=True,
                                           subsampling='4:2:0')
                    
                    compressed_images.append(temp_path)
                    self.progress.emit(int((i + 1) / total_images * 100))
                
                # Combine images back into PDF
                if compressed_images:
                    first_image = Image.open(compressed_images[0])
                    remaining_images = [Image.open(img) for img in compressed_images[1:]]
                    first_image.save(self.output_path, 'PDF', 
                                   save_all=True, 
                                   append_images=remaining_images)
                    
                    self.finished.emit(True, "PDF compression completed successfully!")
                else:
                    raise Exception("No images were processed")
                
        except Exception as e:
            logging.error(f"Compression error: {str(e)}")
            self.finished.emit(False, f"Error during compression: {str(e)}")

class DropArea(QWidget):
    """Custom widget to handle drag and drop of PDF files"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        self.label = QLabel("Drag and drop PDF here\nor click to select file")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].path().lower().endswith('.pdf'):
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.file_dropped.emit(file_path)
        
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.file_dropped.emit(file_path)

class PDFCompressorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file_path = None  # Store the input file path
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('PDF Compressor')
        self.setMinimumSize(600, 400)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Drop area
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self.handle_file_selected)  # Renamed from handle_file_dropped
        layout.addWidget(self.drop_area)
        
        # Selected file display
        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_label)
        
        # Compression settings
        settings_group = QGroupBox("Compression Settings")
        settings_layout = QVBoxLayout()
        
        # Advanced options for scanned documents
        advanced_layout = QHBoxLayout()
        self.optimize_scans_cb = QCheckBox("Optimize for Scanned Documents")
        self.optimize_scans_cb.setToolTip("Enable additional optimizations for scanned documents")
        advanced_layout.addWidget(self.optimize_scans_cb)
        
        self.grayscale_cb = QCheckBox("Convert to Grayscale")
        self.grayscale_cb.setToolTip("Convert images to grayscale to reduce file size")
        advanced_layout.addWidget(self.grayscale_cb)
        
        settings_layout.addLayout(advanced_layout)
        
        # Compression mode selection
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.percentage_radio = QRadioButton("Percentage")
        self.percentage_radio.setChecked(True)
        self.mode_group.addButton(self.percentage_radio)
        mode_layout.addWidget(self.percentage_radio)
        
        self.size_radio = QRadioButton("Target Size (MB)")
        self.mode_group.addButton(self.size_radio)
        mode_layout.addWidget(self.size_radio)
        
        self.resolution_radio = QRadioButton("Resolution (DPI)")
        self.mode_group.addButton(self.resolution_radio)
        mode_layout.addWidget(self.resolution_radio)
        
        settings_layout.addLayout(mode_layout)
        
        # Value input
        value_layout = QHBoxLayout()
        self.value_input = QSpinBox()
        self.value_input.setRange(1, 1000)
        self.value_input.setValue(80)  # Default compression percentage
        value_layout.addWidget(QLabel("Value:"))
        value_layout.addWidget(self.value_input)
        settings_layout.addLayout(value_layout)
        
        # Connect radio buttons to update value range
        self.percentage_radio.toggled.connect(lambda: self.update_value_range('percentage'))
        self.size_radio.toggled.connect(lambda: self.update_value_range('size'))
        self.resolution_radio.toggled.connect(lambda: self.update_value_range('resolution'))
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Output path selection
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output location...")
        output_layout.addWidget(self.output_path)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_output_path)
        output_layout.addWidget(browse_button)
        layout.addLayout(output_layout)
        
        # Compress button
        self.compress_button = QPushButton("Compress PDF")
        self.compress_button.clicked.connect(self.start_compression)
        self.compress_button.setEnabled(False)  # Disabled until file is selected
        layout.addWidget(self.compress_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
        
    def update_value_range(self, mode):
        if mode == 'percentage':
            self.value_input.setRange(1, 100)
            self.value_input.setValue(80)
            self.value_input.setSuffix("%")
        elif mode == 'size':
            self.value_input.setRange(1, 1000)
            self.value_input.setValue(10)
            self.value_input.setSuffix(" MB")
        else:  # resolution
            self.value_input.setRange(72, 600)
            self.value_input.setValue(150)
            self.value_input.setSuffix(" DPI")
            
    def select_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Compressed PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.output_path.setText(file_path)
            
    def handle_file_selected(self, file_path):
        self.input_file_path = file_path
        self.file_label.setText(f"Selected file: {os.path.basename(file_path)}")
        
        if not self.output_path.text():
            # Generate default output path
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            self.output_path.setText(os.path.join(directory, f"{name}_compressed{ext}"))
        
        # Enable compress button
        self.compress_button.setEnabled(True)
            
    def start_compression(self):
        if not self.input_file_path:
            QMessageBox.warning(self, "Error", "Please select a PDF file first.")
            return
            
        if not self.output_path.text():
            QMessageBox.warning(self, "Error", "Please select an output location.")
            return
            
        # Get compression settings
        compression_settings = {
            'mode': 'percentage' if self.percentage_radio.isChecked() else
                   'size' if self.size_radio.isChecked() else 'resolution',
            'value': self.value_input.value(),
            'optimize_scans': self.optimize_scans_cb.isChecked(),
            'grayscale': self.grayscale_cb.isChecked()
        }
        
        # Start compression in worker thread
        self.progress_bar.setVisible(True)
        self.status_label.setText("Compressing PDF...")
        self.compress_button.setEnabled(False)  # Disable during compression
        
        self.worker = PDFCompressorWorker(self.input_file_path, self.output_path.text(), compression_settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.compression_finished)
        self.worker.start()
        
    def compression_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        self.compress_button.setEnabled(True)  # Re-enable button
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
            
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    app = QApplication(sys.argv)
    window = PDFCompressorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PDFCompressorApp()
    window.show()
    sys.exit(app.exec())
else:
    # For use as a module
    __all__ = ['PDFCompressorApp', 'QApplication']