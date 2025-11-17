"""
Generate PolyCouncil.ico from the create_app_icon function
"""
import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Import the icon creation function
from council import create_app_icon

def save_icon_as_ico():
    """Generate and save the icon as .ico file with multiple resolutions"""
    app = QtWidgets.QApplication(sys.argv)  # Required for QIcon operations
    
    # Create icon - we'll generate multiple sizes
    ico_path = "PolyCouncil.ico"
    
    # Try using PIL/Pillow for proper multi-resolution .ico support
    try:
        from PIL import Image
        
        # Generate icons at multiple sizes
        sizes = [16, 32, 48, 64, 128, 256]
        images = []
        
        for size in sizes:
            icon = create_app_icon(size)
            pixmap = icon.pixmap(size, size)
            # Convert QPixmap to PIL Image
            qimage = pixmap.toImage()
            width = qimage.width()
            height = qimage.height()
            # Convert QImage to bytes, then to numpy array
            qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            ptr = qimage.constBits()
            arr = np.array(ptr).reshape((height, width, 4))
            # Convert to PIL Image
            pil_image = Image.fromarray(arr, 'RGBA')
            images.append(pil_image)
        
        # Save as multi-resolution .ico
        # Create ICO file with all sizes - PIL needs special handling
        try:
            # Method 1: Try saving with all images
            ico_file = open(ico_path, 'wb')
            # Write ICO header
            ico_file.write(b'\x00\x00')  # Reserved
            ico_file.write(b'\x01\x00')  # Type (1 = ICO)
            ico_file.write(len(images).to_bytes(2, 'little'))  # Number of images
            
            # Calculate offset for image data
            header_size = 6 + (16 * len(images))  # 6 bytes header + 16 bytes per image entry
            offset = header_size
            
            # Write directory entries
            image_data = []
            for img in images:
                # Convert to BMP format for ICO (ICO files contain BMP data)
                bmp_data = img.tobytes('raw', 'BGRA')
                bmp_size = len(bmp_data) + 40  # BMP header size
                
                # Write directory entry (16 bytes)
                width = img.width if img.width < 256 else 0
                height = img.height if img.height < 256 else 0
                ico_file.write(width.to_bytes(1, 'little'))
                ico_file.write(height.to_bytes(1, 'little'))
                ico_file.write(b'\x00')  # Color palette (0 = no palette)
                ico_file.write(b'\x00')  # Reserved
                ico_file.write(b'\x01\x00')  # Color planes
                ico_file.write((32).to_bytes(2, 'little'))  # Bits per pixel (32 = RGBA)
                ico_file.write(len(bmp_data).to_bytes(4, 'little'))  # Image size
                ico_file.write(offset.to_bytes(4, 'little'))  # Offset to image data
                offset += len(bmp_data) + 40
                image_data.append(bmp_data)
            
            # Write BMP headers and image data
            for i, bmp_data in enumerate(image_data):
                img = images[i]
                # Write BMP header (40 bytes)
                ico_file.write((40).to_bytes(4, 'little'))  # Header size
                ico_file.write(img.width.to_bytes(4, 'little', signed=True))
                ico_file.write((img.height * 2).to_bytes(4, 'little', signed=True))  # Height * 2 for ICO
                ico_file.write((1).to_bytes(2, 'little'))  # Color planes
                ico_file.write((32).to_bytes(2, 'little'))  # Bits per pixel
                ico_file.write((0).to_bytes(4, 'little'))  # Compression (0 = none)
                ico_file.write(len(bmp_data).to_bytes(4, 'little'))  # Image size
                ico_file.write((0).to_bytes(4, 'little'))  # X pixels per meter
                ico_file.write((0).to_bytes(4, 'little'))  # Y pixels per meter
                ico_file.write((0).to_bytes(4, 'little'))  # Colors used
                ico_file.write((0).to_bytes(4, 'little'))  # Important colors
                # Write image data (BGRA, bottom-to-top)
                for y in range(img.height - 1, -1, -1):
                    row_start = y * img.width * 4
                    row_end = row_start + img.width * 4
                    ico_file.write(bmp_data[row_start:row_end])
            
            ico_file.close()
            print(f"Multi-resolution icon saved to {ico_path} (sizes: {[img.size for img in images]})")
        except Exception as e:
            print(f"Error creating ICO manually: {e}, using PIL fallback")
            # Fallback to PIL method
            images[0].save(ico_path, format='ICO')
            print(f"Single-resolution icon saved to {ico_path}")
        
    except ImportError:
        # Fallback: use Qt's built-in .ico support (single resolution)
        print("PIL/Pillow not available, using single-resolution icon")
        icon = create_app_icon(256)
        pixmap = icon.pixmap(256, 256)
        pixmap.save(ico_path, 'ICO')
        print(f"Single-resolution icon saved to {ico_path}")
    except Exception as e:
        # Final fallback
        print(f"Error with PIL method: {e}, using Qt fallback")
        icon = create_app_icon(256)
        pixmap = icon.pixmap(256, 256)
        pixmap.save(ico_path, 'ICO')
        print(f"Icon saved to {ico_path}")
    
    return ico_path

if __name__ == "__main__":
    save_icon_as_ico()

