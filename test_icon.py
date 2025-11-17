from PIL import Image
import os

ico_path = 'PolyCouncil.ico'
if os.path.exists(ico_path):
    img = Image.open(ico_path)
    print(f'Icon format: {img.format}')
    if hasattr(img, 'info') and 'sizes' in img.info:
        print(f'Icon sizes: {img.info["sizes"]}')
    print(f'Icon mode: {img.mode}')
    print(f'Icon size: {img.size}')
    print('Icon file is valid!')
else:
    print('Icon file not found!')




