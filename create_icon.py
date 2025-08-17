import os
from PIL import Image

def create_icns(png_path, icon_path):
    """Convert PNG to ICNS file"""
    # Create temporary directory for iconset
    iconset_path = os.path.splitext(icon_path)[0] + '.iconset'
    os.makedirs(iconset_path, exist_ok=True)

    # Define icon sizes
    sizes = [(16,16), (32,32), (64,64), (128,128), (256,256), (512,512), (1024,1024)]
    
    # Create icons for each size
    for size in sizes:
        img = Image.open(png_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save normal size
        icon_name = f'icon_{size[0]}x{size[0]}.png'
        img.save(os.path.join(iconset_path, icon_name))
        
        # Save @2x size
        icon_name = f'icon_{size[0]//2}x{size[0]//2}@2x.png'
        img.save(os.path.join(iconset_path, icon_name))

    # Convert iconset to icns using iconutil
    os.system(f'iconutil -c icns {iconset_path}')
    
    # Clean up iconset directory
    os.system(f'rm -rf {iconset_path}')

if __name__ == '__main__':
    # Replace with your PNG file path
    png_path = 'images.png'
    icon_path = 'PDFCompressor.icns'
    create_icns(png_path, icon_path)