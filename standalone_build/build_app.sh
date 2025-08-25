#!/bin/bash

# PDF Compressor - Standalone App Builder for Mac Silicon
# This script builds a standalone .app bundle that can be distributed

set -e  # Exit on any error

echo "🔧 PDF Compressor Standalone App Builder"
echo "========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up paths
BUILD_DIR="$SCRIPT_DIR"
DIST_DIR="$BUILD_DIR/dist"
WORK_DIR="$BUILD_DIR/build"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Build directory: $BUILD_DIR"
echo "📁 Distribution directory: $DIST_DIR"

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/pdf_compressor.py" ]; then
    echo "❌ Error: pdf_compressor.py not found in parent directory"
    echo "   Make sure this script is in the standalone_build folder"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"

# Set up virtual environment
VENV_DIR="$BUILD_DIR/build_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" pyinstaller

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "$DIST_DIR" "$WORK_DIR"

# Create necessary directories
mkdir -p "$BUILD_DIR"

# Change to build directory
cd "$BUILD_DIR"

# Check for external dependencies and provide installation guidance
echo "🔍 Checking external dependencies..."

MISSING_DEPS=()

# Check for Homebrew tools
if ! command -v pdftoppm &> /dev/null; then
    MISSING_DEPS+=("poppler (for pdftoppm)")
fi

if ! command -v qpdf &> /dev/null; then
    MISSING_DEPS+=("qpdf")
fi

if ! command -v gs &> /dev/null; then
    MISSING_DEPS+=("ghostscript (for gs)")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠️  Optional dependencies missing (app will work but with reduced functionality):"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo "To install all optional dependencies:"
    echo "   brew install poppler qpdf ghostscript"
    echo ""
    read -p "Continue building without these dependencies? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Build cancelled"
        exit 1
    fi
fi

# Build the app
echo "🚀 Building standalone app..."
python -m PyInstaller PDFCompressor.spec --clean --noconfirm

# Check if build was successful
if [ ! -d "$DIST_DIR/PDFCompressor.app" ]; then
    echo "❌ Build failed - PDFCompressor.app not found"
    exit 1
fi

# Get app size
APP_SIZE=$(du -sh "$DIST_DIR/PDFCompressor.app" | cut -f1)
echo "✅ Build completed successfully!"
echo "📦 App size: $APP_SIZE"
echo "📍 Location: $DIST_DIR/PDFCompressor.app"

# Test the app
echo ""
echo "🧪 Testing the app..."
if "$DIST_DIR/PDFCompressor.app/Contents/MacOS/PDFCompressor" --version 2>/dev/null; then
    echo "✅ App launches successfully"
else
    echo "⚠️  App test failed, but this might be normal for GUI apps"
fi

# Provide next steps
echo ""
echo "🎉 Standalone app created successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Test the app: open '$DIST_DIR/PDFCompressor.app'"
echo "   2. Move to Applications: cp -r '$DIST_DIR/PDFCompressor.app' /Applications/"
echo "   3. Or create a DMG for distribution"
echo ""
echo "💡 The app is self-contained and includes all Python dependencies."
echo "   External tools (poppler, qpdf, ghostscript) are bundled if available."

# Offer to open the app
echo ""
read -p "Open the app now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "$DIST_DIR/PDFCompressor.app"
fi

echo "🏁 Build process complete!"
