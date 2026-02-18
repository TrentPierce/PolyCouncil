# PolyCouncil v1.0.0 - First Release 🎉

## 🚀 What is PolyCouncil?

PolyCouncil is a multi-model deliberation engine for LM Studio that runs multiple local LLMs in parallel, gathers their responses, and has them evaluate each other using a shared scoring rubric. After scoring, the models vote on which answer is best, producing a consensus-driven final result.

**Running a single model gives you one perspective. Running a council gives you… perspective squared.**

## ✨ Key Features

### 🎯 Core Functionality
- **Parallel Model Execution**: Run several local models simultaneously and collect answers in parallel
- **Rubric-Based Scoring**: Each model scores every other response across customizable criteria
- **Consensus Voting System**: Final answer is chosen through a weighted vote based on rubric scores
- **Leaderboard Tracking**: Track which models perform best over time

### 🎭 Persona System
- **Custom Personas**: Create and assign different personas to different models
- **Built-in Personas**: Includes recommended personas like "Meticulous fact-checker", "Pragmatic engineer", "Cautious risk assessor", and more
- **Per-Model Assignment**: Apply different personas to different models simultaneously to test varying perspectives

### ⚖️ Judge Mode
- **Single-Voter Mode**: Optionally use a single model to act as the "ultimate judge" for all votes
- **Flexible Voting**: Choose which models participate in voting vs. answering

### 🔧 Advanced Features
- **Concurrent Processing**: Adjustable concurrency for faster processing (1-8 concurrent jobs)
- **Debug Tools**: Toggle verbose logging for internal scoring/behavior analysis
- **Settings Management**: Centralized settings dialog for easy configuration
- **Model Discovery**: Automatic detection of available models from LM Studio

## 📦 What's New in v1.0.0

### 🎁 Standalone Windows Executable
- **No Python Required**: Download and run `PolyCouncil.exe` directly - no installation needed!
- **All Dependencies Included**: Everything bundled in a single executable (~36 MB)
- **Custom Icon**: Beautiful blue icon matching the application theme

### 🎨 UI Improvements
- **Settings Dialog**: Centralized settings with persona management
- **Persona Buttons**: Easy-to-use persona selection buttons for each model
- **Debug Log Dock**: Dockable debug log panel for detailed analysis
- **Modern Interface**: Clean, intuitive GUI built with PySide6

### 🐛 Bug Fixes & Improvements
- Fixed persona system visibility and functionality
- Improved model list refresh stability
- Enhanced icon display on Windows
- Better error handling and user feedback

## 📋 System Requirements

- **Operating System**: Windows 10/11 (64-bit)
- **LM Studio**: Must be installed and running with at least one model loaded
- **RAM**: Recommended 8GB+ (depends on your models)
- **Storage**: ~50 MB for the executable

**Note**: LM Studio must be running and accessible at `http://localhost:1234` (default) or your configured base URL.

## 🚀 Quick Start

1. **Download**: Download `PolyCouncil.exe` from this release
2. **Run LM Studio**: Start LM Studio and load at least one model
3. **Launch**: Double-click `PolyCouncil.exe`
4. **Connect**: Click "Connect" to discover available models
5. **Select Models**: Check the models you want to use
6. **Ask Questions**: Type your question and click "Send"

## 📖 Usage Tips

### First Time Setup
1. Open Settings (top-right button) to configure:
   - Enable personas if you want to use different perspectives
   - Adjust max concurrency based on your hardware
   - Enable debug logs for detailed analysis

### Using Personas
1. Go to Settings → Enable personas
2. Click the "Persona" button next to each model
3. Select a persona from the menu
4. Different personas will give different perspectives on the same question

### Single-Voter Mode
- Check "Single-voter" in the main window
- Select which model should act as the judge
- All models will answer, but only the selected model will vote

### Debug Mode
- Enable in Settings → "Enable debug logs"
- A dockable log panel will show detailed scoring and voting information
- Useful for understanding how the council reaches consensus

## 🎯 Use Cases

- **Model Benchmarking**: Compare how different models perform on the same questions
- **Research**: Explore emergent behavior from model voting and deliberation
- **Reliability**: Get more reliable answers by combining multiple model perspectives
- **Experimentation**: Test different personas and voting strategies
- **Education**: Understand how ensemble methods work in practice

## ⚠️ Known Issues & Notes

- **First Run**: Windows Defender may scan the executable on first launch (this is normal)
- **Icon Cache**: If the icon doesn't display correctly, Windows may need to refresh its icon cache (restart File Explorer)
- **LM Studio Connection**: Ensure LM Studio is running before connecting
- **Model Loading**: Large models may take time to respond - be patient!

## 🔗 Alternative Installation

Prefer to run from source? Clone the repository and install dependencies:

```bash
git clone https://github.com/TrentPierce/PolyCouncil
cd PolyCouncil
pip install -r requirements.txt
python council.py
```

## 📄 License

This project is licensed under the **Polyform Noncommercial License 1.0.0**.
- ✅ Free for personal and research use
- ❌ Commercial use is not permitted

## 🙏 Acknowledgments

Built for model tinkerers, curious minds, and anyone who enjoys watching artificial brains argue politely.

## 🐛 Report Issues

Found a bug or have a feature request? Please open an issue on GitHub:
https://github.com/TrentPierce/PolyCouncil/issues

---

**Enjoy exploring the power of multi-model deliberation!** 🎉




