# Fixing Icon Display in PolyCouncil EXE

## The Problem

PyInstaller's icon embedding is unreliable on Windows. Even though it reports "Copying icon to EXE", Windows may still show the default icon.

## The Solution: Use rcedit

The most reliable way to embed icons in Windows EXEs is using `rcedit`, a Node.js tool.

### Installation

1. Install Node.js from https://nodejs.org/
2. Install rcedit globally:
   ```bash
   npm install -g rcedit
   ```

### Usage

After building the EXE, run:
```bash
rcedit dist\PolyCouncil.exe --set-icon PolyCouncil.ico
```

Or use the automated build script:
```bash
build_with_icon.bat
```

## Alternative: Manual Icon Embedding

If you can't use rcedit, you can:

1. Build the EXE normally
2. Use a tool like Resource Hacker (free) to manually add the icon:
   - Download Resource Hacker from http://www.angusj.com/resourcehacker/
   - Open `dist\PolyCouncil.exe` in Resource Hacker
   - Go to Action > Add a Resource
   - Select `PolyCouncil.ico`
   - Save the modified EXE

## Verification

After embedding the icon:
1. Clear icon cache: `clear_icon_cache.bat`
2. Refresh the folder (F5)
3. The icon should now display correctly




