# Building PolyCouncil EXE

This guide explains how to build a standalone executable for PolyCouncil.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Building the EXE

### Option 1: Using the build script (Windows)

1. Open a command prompt in the project directory
2. Run:
   ```
   build.bat
   ```

### Option 2: Manual build

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Build the EXE:
   ```
   pyinstaller build_exe.spec --clean
   ```

3. The executable will be created in the `dist` folder as `PolyCouncil.exe`

## Distribution

After building, you can distribute the `PolyCouncil.exe` file from the `dist` folder. The executable is standalone and includes all necessary dependencies.

**Note:** The first run may be slightly slower as Windows Defender scans the new executable.

## Troubleshooting

- If the build fails, make sure all dependencies are installed: `pip install -r requirements.txt`
- If the EXE doesn't run, try building with `--debug=all` to see error messages
- For smaller file size, you can remove `upx=True` from the spec file (but UPX compression helps reduce size)

## File Size

The resulting EXE will be approximately 50-100 MB due to included Qt libraries and Python runtime. This is normal for PySide6 applications.




