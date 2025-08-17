# PDF Compressor

A powerful PDF compression tool with a modern GUI that intelligently reduces PDF file sizes while preserving quality. Features multiple compression strategies including vector-preserving compression, selective rasterization, and grayscale conversion.

## Features

- **Multiple Compression Modes**: Percentage-based, target file size, DPI reduction, and size reduction
- **Intelligent Compression**: Vector-preserving by default with selective rasterization when needed
- **Quality Presets**: Pre-configured settings for common use cases (email-safe, web-view, high fidelity, etc.)
- **Advanced Options**: Grayscale conversion, color fidelity preservation, custom DPI and JPEG quality floors
- **Real-time Analysis**: Automatic PDF analysis to determine optimal compression strategy
- **Progress Monitoring**: Live progress updates and memory usage tracking
- **Drag & Drop Interface**: Simple file selection via drag and drop or file browser

## Requirements

### Dependencies
- Python 3.8+
- PyQt6 (GUI framework)
- PyMuPDF (PDF processing)
- pypdf (PDF analysis)
- Pillow (image processing)
- psutil (system monitoring)

### External Tools (Optional but Recommended)
- **Poppler** (pdftoppm, pdfimages) - Enhanced image analysis
- **qpdf** - PDF linearization for web optimization
- **Ghostscript** - Specialized compression for B/W scanned documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PDFCompressor.git
cd PDFCompressor
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install external tools (optional):

**macOS (using Homebrew):**
```bash
brew install poppler qpdf ghostscript
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils qpdf ghostscript
```

**Windows:**
- Download and install from respective official websites
- Ensure tools are in your system PATH

## Usage

### GUI Application
```bash
python pdf_compressor.py
```

1. Launch the application
2. Drag and drop a PDF file or click to browse
3. Select a compression preset or customize settings
4. Choose output location
5. Click "Compress PDF"

### Compression Presets

- **Manual/Custom**: Customizable settings (50% compression by default)
- **Sharable Doc (1 MB)**: Grayscale, optimized for document sharing
- **Email-safe (0.5 MB)**: Aggressive compression for email attachments
- **Web-view (50%)**: Balanced compression for web display
- **High fidelity (75%)**: Minimal compression with color preservation
- **Smallest reasonable (35%)**: Maximum compression while maintaining readability
- **Presentation images (20 MB)**: Large file size with high image quality

### Advanced Settings

- **Min DPI**: Minimum resolution floor for rasterized content (50-300 DPI)
- **Min JPEG Quality**: Minimum JPEG compression quality (10-90)
- **Convert to grayscale**: Force grayscale conversion for smaller files
- **Force rasterize if needed**: Allow full document rasterization to meet targets
- **Preserve color fidelity**: Use higher quality JPEG encoding (4:4:4 vs 4:2:0)

## How It Works

The compressor uses a multi-stage approach:

1. **PDF Analysis**: Analyzes document structure, text content, and image types
2. **In-place Optimization**: Recompresses existing images without rasterization
3. **Selective Rasterization**: Converts only heavy pages to images when needed
4. **Full Rasterization**: Complete document rasterization as last resort
5. **Specialized Handling**: Uses Ghostscript for B/W scanned documents

## File Structure

```
PDFCompressor/
├── pdf_compressor.py          # Main application
├── requirements.txt           # Python dependencies
├── create_icon.py            # Icon creation utility
├── images.png                # Application icon source
├── Info.plist               # macOS app bundle info
├── PDFCompressorBuild/       # Build artifacts and distribution files
└── venv/                     # Virtual environment (if present)
```

## Building Standalone Application

The project includes PyInstaller configuration for creating standalone executables:

```bash
pyinstaller PDFCompressorBuild/PDFCompressor.spec
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyQt6 for the GUI framework
- Uses PyMuPDF for PDF processing and rendering
- Leverages Poppler tools for enhanced PDF analysis
- Ghostscript integration for specialized compression scenarios