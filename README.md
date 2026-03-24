# PolyCouncil

<div align="center">

**A QML desktop app for multi-model deliberation and discussion**

*Compare local and hosted models in one workflow, stream responses live, and keep the full decision trail.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Polyform%20Noncommercial%201.0.0-yellow?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Desktop-0078D4?style=for-the-badge)](https://github.com/TrentPierce/PolyCouncil)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/TrentPierce/PolyCouncil)

[![GitHub stars](https://img.shields.io/github/stars/TrentPierce/PolyCouncil?style=social)](https://github.com/TrentPierce/PolyCouncil/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/TrentPierce/PolyCouncil?style=social)](https://github.com/TrentPierce/PolyCouncil/network/members)
[![GitHub issues](https://img.shields.io/github/issues/TrentPierce/PolyCouncil?color=orange)](https://github.com/TrentPierce/PolyCouncil/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/TrentPierce/PolyCouncil?color=brightgreen)](https://github.com/TrentPierce/PolyCouncil/pulls)

</div>

## Overview

PolyCouncil coordinates multiple LLMs across local and hosted providers, then lets you review the run as a structured workflow instead of a loose stack of chat outputs.

The current desktop app uses:
- a QML primary UI in `gui/main.qml`
- a Python bridge in `qml_bridge.py`
- a clean runtime entrypoint in `main.py`

The legacy widgets-era `council.py` is still in the repo for compatibility and reference, but it is no longer the primary app entrypoint.

## Core Modes

### Deliberation

Multiple models answer the same prompt, score peer answers, and produce a weighted winner.

### Discussion

Multiple models collaborate across turns and generate a final synthesis.

## Features

### Providers

- LM Studio
- Ollama
- OpenAI-compatible hosted APIs
- OpenAI
- OpenRouter
- Google Gemini
- Anthropic
- Groq
- Together AI
- Kimi
- MiniMax
- Z.AI
- Fireworks AI

### Workflow

- provider profiles and saved connection presets
- searchable large model lists
- persona assignment and persona editing
- file attachments and image attachments
- live streaming output when the provider supports it
- session replay and JSON export
- leaderboard/history tracking

### Safety and persistence

- sanitized markdown rendering
- secure API key storage when the OS keychain is available
- session history stored under app data directories

## Screenshots

### Main Workflow

![PolyCouncil main workflow](docs/app-main-workflow.png)

### Settings And Persona Library

![PolyCouncil settings and persona library](docs/app-settings-personas.png)

### Results Overview

![PolyCouncil results overview](docs/app-results-overview.png)

## Quick Start

### Windows EXE

1. Download the latest `PolyCouncil.exe` from [Releases](https://github.com/TrentPierce/PolyCouncil/releases).
2. Start LM Studio or Ollama, or prepare a hosted provider API key.
3. Launch `PolyCouncil.exe`.
4. Load models from the left workflow pane.
5. Select models, write a prompt, and run the council.

### Run from source

```bash
git clone https://github.com/TrentPierce/PolyCouncil.git
cd PolyCouncil
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.11+
- `PySide6`
- `aiohttp`
- `qasync`
- `pypdf`
- `python-docx`
- `markdown`
- `bleach`
- `keyring`

## Source layout

- `main.py`: QML app entrypoint
- `qml_bridge.py`: QObject bridge exposed to QML
- `gui/main.qml`: main application shell
- `gui/components/`: reusable QML controls
- `core/`: provider routing, discussion logic, sessions, settings, rendering helpers
- `tests/`: pytest coverage for providers, voting, personas, discussion flow, and rendering safety

## Build

### Build the desktop executable

```bash
python -m PyInstaller build_exe.spec --clean --noconfirm
```

Output:

- `dist/PolyCouncil.exe` on Windows
- `dist/PolyCouncil` on Linux and macOS runners

The build spec targets `main.py` and bundles the QML UI, QML components, personas config, and app icon.

## Development

```bash
git clone https://github.com/TrentPierce/PolyCouncil.git
cd PolyCouncil
pip install -r requirements-dev.txt
python main.py
```

Run tests:

```bash
pytest -q
```

## Data locations

### Settings, profiles, and user personas

- Windows: `%APPDATA%\\PolyCouncil`
- macOS: `~/Library/Application Support/PolyCouncil`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/PolyCouncil`

### Leaderboard and session history

- Windows: `%LOCALAPPDATA%\\PolyCouncil`
- macOS: `~/Library/Application Support/PolyCouncil`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/PolyCouncil`

## Troubleshooting

### Models do not load

- verify the provider base URL and API key
- confirm the provider actually exposes a compatible model-list endpoint
- use manual model entry for providers that do not expose a standard list endpoint

### OpenRouter privacy errors

Some OpenRouter models, especially `:free` routes, may fail if your account privacy settings do not allow any available backend for that model. Adjust the privacy policy in your OpenRouter account or choose a different route.

### No winner selected

Failed or cancelled answers are excluded from the voting pool. If every model fails, PolyCouncil will finish the run without forcing a winner.

## Contributing

Contributions are welcome in:
- provider compatibility
- voting quality
- streaming behavior
- UI polish
- documentation
- test coverage

Open an issue or submit a pull request.

## License

This project is licensed under the **Polyform Noncommercial License 1.0.0**.

See [LICENSE](LICENSE) for details.

## Release Notes

- [v1.1.1](docs/releases/RELEASE_NOTES_v1.1.1.md)
- [v1.1.0](docs/releases/RELEASE_NOTES_v1.1.0.md)
- [v1.0.1](docs/releases/RELEASE_NOTES_v1.0.1.md)
- [v1.0.0](docs/releases/RELEASE_NOTES_v1.0.0.md)

## Support

- Bug reports: [GitHub Issues](https://github.com/TrentPierce/PolyCouncil/issues)
- Feature requests: [GitHub Issues](https://github.com/TrentPierce/PolyCouncil/issues)
- Project page: [GitHub Repository](https://github.com/TrentPierce/PolyCouncil)
