#!/bin/bash

# PDF Compressor - Distribution Package Creator
# Creates a DMG file for easy distribution of the standalone app

set -e

echo "📦 PDF Compressor Distribution Package Creator"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$SCRIPT_DIR/dist/PDFCompressor.app"
DMG_NAME="PDFCompressor_v1.0_MacSilicon.dmg"
VOLUME_NAME="PDF Compressor"

# Check if app exists
if [ ! -d "$APP_PATH" ]; then
    echo "❌ Error: PDFCompressor.app not found at $APP_PATH"
    echo "   Please run ./build_app.sh first to build the app"
    exit 1
fi

# Clean up any existing DMG
if [ -f "$SCRIPT_DIR/$DMG_NAME" ]; then
    echo "🗑️  Removing existing DMG..."
    rm "$SCRIPT_DIR/$DMG_NAME"
fi

# Get app size for DMG sizing
APP_SIZE=$(du -sk "$APP_PATH" | cut -f1)
# Add 50MB buffer (in KB)
DMG_SIZE=$((APP_SIZE + 50000))

echo "📊 App size: $(du -sh "$APP_PATH" | cut -f1)"
echo "📊 DMG size: $((DMG_SIZE / 1024))MB"

# Create temporary DMG
echo "🔨 Creating DMG..."
TEMP_DMG="$SCRIPT_DIR/temp.dmg"

hdiutil create -srcfolder "$APP_PATH" \
    -volname "$VOLUME_NAME" \
    -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" \
    -format UDRW \
    -size ${DMG_SIZE}k \
    "$TEMP_DMG"

# Mount the DMG
echo "🔗 Mounting DMG for customization..."
MOUNT_DIR=$(mktemp -d)
hdiutil attach "$TEMP_DMG" -readwrite -mountpoint "$MOUNT_DIR"

# Create Applications symlink
echo "🔗 Creating Applications symlink..."
ln -s /Applications "$MOUNT_DIR/Applications"

# Set up the appearance (optional - basic setup)
echo "🎨 Setting up DMG appearance..."

# Create a basic .DS_Store for positioning (if needed)
# This is optional - the DMG will work fine without it

# Unmount the DMG
echo "📤 Unmounting DMG..."
hdiutil detach "$MOUNT_DIR"

# Convert to compressed read-only DMG
echo "🗜️  Compressing DMG..."
hdiutil convert "$TEMP_DMG" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$SCRIPT_DIR/$DMG_NAME"

# Clean up temp DMG
rm "$TEMP_DMG"

# Get final DMG size
FINAL_SIZE=$(du -sh "$SCRIPT_DIR/$DMG_NAME" | cut -f1)

echo "✅ DMG created successfully!"
echo "📦 File: $SCRIPT_DIR/$DMG_NAME"
echo "📊 Size: $FINAL_SIZE"
echo ""
echo "🚀 Distribution ready!"
echo "   Users can download and mount the DMG, then drag the app to Applications"
echo ""
echo "💡 The DMG contains:"
echo "   - PDFCompressor.app (standalone app)"
echo "   - Applications symlink (for easy installation)"

# Ask if user wants to open the DMG
read -p "Open the DMG to test? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "$SCRIPT_DIR/$DMG_NAME"
fi

echo "🏁 Distribution package complete!"
