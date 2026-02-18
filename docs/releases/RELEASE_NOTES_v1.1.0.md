# PolyCouncil v1.1.0 Release Notes

**Release Date:** January 20, 2026

---

## 🎉 What's New

### Stability & Reliability Improvements

- **Thread-Safe Settings**: Fixed a race condition that could cause settings to be lost when toggling multiple options quickly
- **Single-Voter Validation**: The app now verifies that the selected single-voter model is actually loaded before using it, with a helpful warning if not
- **Better Error Messages**: Clearer feedback when issues occur, especially around model compatibility and voting

### Fairness Improvements

- **Fair Tie-Breaking**: When multiple models score equally, the winner is now randomly selected instead of always picking the first one. This eliminates positional bias that gave an unfair advantage to models selected earlier.
- **Smarter Consensus Detection**: Discussion mode now actually analyzes message content for agreement signals (words like "agree", "consensus", "conclude") rather than just counting turns

### User Experience

- **Keyboard Shortcuts**: 
  - `Ctrl+Enter` - Send message to council
  - `Ctrl+Shift+A` - Select all models
  - `Ctrl+R` - Refresh models list  
  - `Escape` - Stop current operation
  
- **Cleaner UI**: The concurrency warning only appears when you set concurrency above 2 (no more unnecessary clutter)

- **Enhanced Leaderboard**: Now shows win percentages alongside win counts (e.g., "ModelName — 5 wins (42%)")

### Bug Fixes

- Fixed `stat_view.py` which had a hardcoded path that only worked on the developer's machine
- Fixed PySide6 compatibility issue with keyboard shortcuts

---

## 📦 Installation

### Standalone Executable (Recommended)
1. Download `PolyCouncil.exe` from this release
2. Run the exe (no installation required)
3. Make sure LM Studio is running with at least one model loaded
4. Click "Connect" and start deliberating!

### From Source
```bash
git clone https://github.com/TrentPierce/PolyCouncil.git
cd PolyCouncil
pip install -r requirements.txt
python council.py
```

---

## 🙏 Acknowledgments

Thanks to everyone who provided feedback and reported issues. This release addresses the findings from a comprehensive code audit focused on stability, fairness, and user experience.

---

**Full Changelog**: https://github.com/TrentPierce/PolyCouncil/compare/v1.0.1...v1.1.0
