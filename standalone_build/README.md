# PDF Compressor - Standalone Build

This directory contains everything needed to build a standalone Mac Silicon app for PDF Compressor.

## Files in this directory:

- `build_app.sh` - Main build script
- `PDFCompressor.spec` - PyInstaller specification file
- `bootstrap.py` - Application bootstrap/launcher
- `qt.conf` - Qt configuration for proper plugin loading

## Building the App

### Prerequisites

1. **Python 3.8+** with the required packages:
   ```bash
   pip install -r ../requirements.txt
   pip install pyinstaller
   ```

2. **Optional external tools** (recommended for full functionality):
   ```bash
   brew install poppler qpdf ghostscript
   ```

### Build Process

1. Make sure you're in the project root directory
2. Run the build script:
   ```bash
   ./standalone_build/build_app.sh
   ```

The script will:
- Check for dependencies
- Clean previous builds
- Create a standalone .app bundle
- Test the app
- Provide next steps

### Output

The built app will be located at:
```
standalone_build/dist/PDFCompressor.app
```

## Distribution

To distribute the app:

1. **Simple copy**: Move to Applications folder
   ```bash
   cp -r standalone_build/dist/PDFCompressor.app /Applications/
   ```

2. **Create DMG** (for distribution):
   ```bash
   hdiutil create -volname "PDF Compressor" -srcfolder standalone_build/dist/PDFCompressor.app -ov -format UDZO PDFCompressor.dmg
   ```

## Notes

- The app is built specifically for Mac Silicon (ARM64)
- All Python dependencies are bundled
- External tools (poppler, qpdf, ghostscript) are included if available
- The app is self-contained and doesn't require Python to be installed on the target machine

## Troubleshooting

If the build fails:

1. Check that all Python dependencies are installed
2. Ensure you're running on macOS with Apple Silicon
3. Check the terminal output for specific error messages
4. Try cleaning and rebuilding:
   ```bash
   rm -rf standalone_build/dist standalone_build/build
   ./standalone_build/build_app.sh
   ```

## Keeping Build Separate from Git

This directory and all build artifacts are separate from your main git repository. To keep it that way:

1. Add to your `.gitignore`:
   ```
   standalone_build/dist/
   standalone_build/build/
   *.app
   *.dmg
   ```

2. Only commit the build configuration files, not the built artifacts.
