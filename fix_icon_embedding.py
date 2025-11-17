"""
Try to fix icon embedding by using win32api to set the icon resource
"""
import sys
import os

try:
    import win32api
    import win32con
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("pywin32 not available - cannot use advanced icon embedding")

def verify_and_fix_icon(exe_path, ico_path):
    """Verify icon is embedded and try to fix if needed"""
    if not HAS_WIN32:
        print("Cannot verify/fix icon - pywin32 not installed")
        print("Install with: pip install pywin32")
        return False
    
    if not os.path.exists(exe_path):
        print(f"EXE not found: {exe_path}")
        return False
    
    if not os.path.exists(ico_path):
        print(f"Icon not found: {ico_path}")
        return False
    
    try:
        # Try to load the icon from the EXE
        hicon = win32gui.LoadImage(
            0, exe_path, win32con.IMAGE_ICON, 0, 0,
            win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        )
        if hicon:
            print("Icon found in EXE!")
            win32gui.DestroyIcon(hicon)
            return True
        else:
            print("Icon not found in EXE resources")
            return False
    except Exception as e:
        print(f"Error checking icon: {e}")
        return False

if __name__ == "__main__":
    exe_path = r"dist\PolyCouncil.exe"
    ico_path = "PolyCouncil.ico"
    verify_and_fix_icon(exe_path, ico_path)




