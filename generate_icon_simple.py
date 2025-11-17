"""
Generate PolyCouncil.ico using PIL's built-in ICO support
This should create a more Windows-compatible icon file
"""
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image
import numpy as np

# Import the icon creation function
from council import create_app_icon

def save_icon_as_ico():
    """Generate and save the icon as .ico file with multiple resolutions"""
    app = QtWidgets.QApplication(sys.argv)  # Required for QIcon operations
    
    ico_path = "PolyCouncil.ico"
    
    # Generate icons at multiple sizes (Windows standard sizes)
    sizes = [16, 32, 48, 256]  # Standard Windows icon sizes
    images = []
    
    for size in sizes:
        icon = create_app_icon(size)
        pixmap = icon.pixmap(size, size)
        # Convert QPixmap to PIL Image
        qimage = pixmap.toImage()
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        
        # Get image data
        # QImage Format_RGBA8888 - the byte order depends on endianness
        # On little-endian (Windows), Format_RGBA8888 is stored as BGRA in memory
        byte_data = qimage.constBits()
        # Convert memoryview to bytes, then to numpy array
        arr = np.frombuffer(bytes(byte_data), dtype=np.uint8).reshape((height, width, 4))
        
        # Try both formats - if orange appears, channels are swapped
        # Format_RGBA8888 on little-endian Windows is BGRA
        # Test: if blue (#1f80d6) shows as orange, we need to swap
        # Blue #1f80d6 = RGB(31, 128, 214)
        # If showing orange, it's likely RGB(214, 128, 31) = swapped R and B
        arr_rgba = arr.copy()
        arr_rgba[:, :, [0, 2]] = arr_rgba[:, :, [2, 0]]  # Swap B and R: BGRA -> RGBA
        pil_image = Image.fromarray(arr_rgba, 'RGBA')
        images.append(pil_image)
    
    # Save using PIL's ICO format - this should handle multi-resolution correctly
    # Save the largest first, then append others
    try:
        # PIL's ICO format should handle multiple sizes
        images[0].save(
            ico_path,
            format='ICO',
            sizes=[(img.width, img.height) for img in images]
        )
        print(f"Multi-resolution icon saved to {ico_path}")
        print(f"Sizes: {[img.size for img in images]}")
        
        # Verify the saved file
        verify_img = Image.open(ico_path)
        if hasattr(verify_img, 'info') and 'sizes' in verify_img.info:
            print(f"Verified sizes in ICO: {verify_img.info['sizes']}")
        else:
            print("Warning: Could not verify all sizes in ICO file")
            
    except Exception as e:
        print(f"Error saving ICO: {e}")
        # Fallback: save single size
        images[-1].save(ico_path, format='ICO')
        print(f"Saved single-resolution icon (256x256) to {ico_path}")
    
    return ico_path

if __name__ == "__main__":
    save_icon_as_ico()

