"""
Check if the EXE has an icon embedded
"""
import sys
import os

exe_path = r'dist\PolyCouncil.exe'

if not os.path.exists(exe_path):
    print(f"EXE not found at {exe_path}")
    sys.exit(1)

# Try to read the EXE and look for icon resources
# This is a simple check - Windows stores icons in the resource section
try:
    with open(exe_path, 'rb') as f:
        data = f.read()
        # Look for ICO signature in the file (not perfect but gives an indication)
        ico_sig = b'\x00\x00\x01\x00'  # ICO file signature
        if ico_sig in data:
            print("ICO signature found in EXE - icon may be embedded")
            # Count occurrences
            count = data.count(ico_sig)
            print(f"  Found {count} potential icon resource(s)")
        else:
            print("ICO signature NOT found in EXE")
            print("  The icon may not be properly embedded")
        
        # Check file size
        size_mb = len(data) / (1024 * 1024)
        print(f"\nEXE size: {size_mb:.2f} MB")
        
except Exception as e:
    print(f"Error checking EXE: {e}")

print("\nNote: Windows may cache icons. Try:")
print("1. Refresh the folder (F5)")
print("2. Run clear_icon_cache.bat")
print("3. Check EXE Properties > General tab to see if icon appears there")

