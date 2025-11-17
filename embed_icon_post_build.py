"""
Post-build script to properly embed icon using rcedit or similar tool
"""
import os
import subprocess
import sys

def embed_icon_with_rcedit(exe_path, ico_path):
    """Use rcedit to embed icon (if available)"""
    try:
        # Try to use rcedit (Node.js tool)
        result = subprocess.run(
            ['rcedit', exe_path, '--set-icon', ico_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Icon embedded successfully using rcedit!")
            return True
        else:
            print(f"rcedit failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("rcedit not found. Install with: npm install -g rcedit")
        return False
    except Exception as e:
        print(f"Error using rcedit: {e}")
        return False

def embed_icon_with_python(exe_path, ico_path):
    """Try to embed icon using Python libraries"""
    try:
        import pefile
        # This is complex - would need to modify PE resources
        print("PE file modification not implemented")
        return False
    except ImportError:
        print("pefile not available")
        return False

if __name__ == "__main__":
    exe_path = r"dist\PolyCouncil.exe"
    ico_path = "PolyCouncil.ico"
    
    if not os.path.exists(exe_path):
        print(f"EXE not found: {exe_path}")
        sys.exit(1)
    
    if not os.path.exists(ico_path):
        print(f"Icon not found: {ico_path}")
        sys.exit(1)
    
    print("Attempting to embed icon using rcedit...")
    if embed_icon_with_rcedit(exe_path, ico_path):
        print("Success!")
    else:
        print("\nAlternative: Install rcedit:")
        print("  npm install -g rcedit")
        print("Then run: rcedit dist\\PolyCouncil.exe --set-icon PolyCouncil.ico")




