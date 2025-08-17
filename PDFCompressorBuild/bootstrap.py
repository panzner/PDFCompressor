import os
import sys
import logging
from pathlib import Path

def setup_logging():
    """Set up logging to file and console"""
    log_file = Path.home() / 'pdf_compressor_log.txt'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info('Starting PDF Compressor')
    logging.info(f'Python version: {sys.version}')
    logging.info(f'Current working directory: {os.getcwd()}')
    logging.info(f'sys.path: {sys.path}')

def init_qt_plugins():
    """Initialize Qt plugins and paths"""
    try:
        if getattr(sys, 'frozen', False):
            # Running in a bundle
            bundle_dir = os.path.dirname(sys.executable)
            logging.info(f'Bundle directory: {bundle_dir}')
            
            # Set Qt plugin paths
            os.environ['QT_DEBUG_PLUGINS'] = '1'
            os.environ['QT_PLUGIN_PATH'] = os.path.join(bundle_dir, 'plugins')
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(bundle_dir, 'plugins', 'platforms')
            
            logging.info(f'QT_PLUGIN_PATH: {os.environ.get("QT_PLUGIN_PATH")}')
            logging.info(f'QT_QPA_PLATFORM_PLUGIN_PATH: {os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")}')
            
            # List available plugins
            logging.info('Available plugins:')
            for root, dirs, files in os.walk(os.environ['QT_PLUGIN_PATH']):
                for file in files:
                    logging.info(f'  {os.path.join(root, file)}')
    except Exception as e:
        logging.error(f'Error initializing Qt plugins: {str(e)}')
        raise

def main():
    try:
        # Set up logging first
        setup_logging()
        
        # Initialize Qt
        init_qt_plugins()
        
        # Import and run the main application
        logging.info('Importing enhanced PDF compressor...')
        try:
            from pdf_compressor import PDFCompressorApp, QApplication, DependencyChecker
            
            # Check critical dependencies before starting UI
            logging.info('Checking dependencies...')
            if not DependencyChecker.check_poppler():
                logging.warning('poppler-utils not found. Image-based compression may fail.')
            
            available_memory = DependencyChecker.check_available_memory()
            logging.info(f'Available memory: {available_memory:.1f}GB')
            
            if available_memory < 1:
                logging.warning('Low memory detected. Large PDFs may cause issues.')
            
            logging.info('Creating enhanced application...')
            app = QApplication(sys.argv)
            window = PDFCompressorApp()
            window.show()
            
            logging.info('Starting event loop...')
            sys.exit(app.exec())
            
        except ImportError as e:
            logging.error(f'Failed to import main application: {str(e)}')
            logging.error(f'Current directory contents: {os.listdir(".")}')
            raise
            
    except Exception as e:
        logging.error(f'Fatal error: {str(e)}')
        raise

if __name__ == '__main__':
    main()