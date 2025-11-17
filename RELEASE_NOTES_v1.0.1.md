# PolyCouncil v1.0.1 - Icon Fix Release

## What's New

### Icon Fixes
- **Fixed Icon Color**: Resolved color accuracy issues in the application icon
- **Fixed Icon Scaling**: Improved icon scaling across different Windows display contexts and resolutions
- **Better Icon Quality**: Updated to use optimized icon file for better visual fidelity

## Changes

- Updated EXE build configuration to use `PolyCouncilIco.ico` for improved color accuracy and proper scaling
- Icon now displays correctly with the intended blue color scheme
- Multi-resolution icon support ensures proper display at all sizes

## Installation

1. **Download** `PolyCouncil.exe` from this release
2. **Replace** your existing `PolyCouncil.exe` with the new version
3. **Run** the application - the icon should now display correctly

**Note**: If the icon still shows incorrectly, Windows may have cached the old icon. Try:
- Restarting File Explorer
- Moving the EXE to a different location
- Clearing Windows icon cache

## Technical Details

This release addresses icon rendering issues where the programmatically generated icon had color channel conversion problems and scaling artifacts. The fix uses a pre-optimized icon file that ensures proper color representation and scaling across all Windows display contexts.

## Compatibility

- Same system requirements as v1.0.0
- Windows 10/11 (64-bit)
- LM Studio with at least one model loaded

---

**Enjoy the improved icon display!**

