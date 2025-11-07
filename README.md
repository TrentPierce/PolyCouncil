PolyCouncil

A multi-model deliberation engine for LM Studio.

PolyCouncil runs multiple local LLMs in parallel, gathers their responses, and has them evaluate each other using a shared scoring rubric. After scoring, the models vote on which answer is best, producing a consensus-driven final result.
This creates a more reliable, research-friendly way to test, compare, and orchestrate multiple models inside LM Studio.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Polyform%20Noncommercial%201.0.0-yellow?style=for-the-badge">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-LM%20Studio-4B32FF?style=for-the-badge&logo=data:image/svg+xml;base64,">
  <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/TrentPierce/PolyCouncil?style=for-the-badge&color=brightgreen">
  <img alt="Issues" src="https://img.shields.io/github/issues/TrentPierce/PolyCouncil?style=for-the-badge&color=orange">
  <img alt="Stars" src="https://img.shields.io/github/stars/TrentPierce/PolyCouncil?style=for-the-badge&color=yellow">
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/TrentPierce/PolyCouncil?style=for-the-badge&color=blue">
</p>

Features

Parallel model execution
Run several local models at once and collect answers simultaneously.

Rubric-based scoring
Each model scores every other response across customizable criteria.

Consensus voting system
Final answer is chosen through a weighted vote based on rubric scores.

Judge mode
Optionally use a single model to act as the “ultimate judge.”

Debug tools
Toggle verbose console logging for internal scoring/behavior analysis.

Open source + extensible
Designed for easy tinkering, experimentation, and model comparisons.

Why PolyCouncil?

Running a single model gives you one perspective.
Running a council gives you… perspective squared.

PolyCouncil helps you:

Benchmark local models against each other

Identify strengths and weaknesses between architectures

Explore emergent behavior from model voting

Improve reliability by combining multiple opinions

Build your own research tools on top of a clean ensemble framework

Perfect for hobbyists, researchers, and anyone who likes collecting too many models in LM Studio.

Getting Started
git clone https://github.com/TrentPierce/PolyCouncil
cd PolyCouncil
python main.py


(Replace with your actual entry point if different.)

Requirements

Python 3.10+

LM Studio with local models installed

Packages listed in requirements.txt

Configuration

PolyCouncil supports:

Dynamic model lists

Custom scoring rubrics

Judge-only mode

Adjustable temperature/top_p for each model

Optional debug logging

Check the config/ folder for examples and presets.

Screenshots

Coming soon

Contributing

Pull requests are welcome.
Whether it's fixing a bug, improving the GUI, optimizing concurrency, or adding new scoring mechanisms, contributions help the council grow smarter.

License

Polyform Noncommercial License 1.0.0
Free for personal and research use. Commercial use is not permitted.

Roadmap

Web UI version

More advanced judge agents

Plugin system for custom scoring rules

Model personality profiles / biases testing

Automatic answer explanation generator

Graph-based deliberation visualization

Acknowledgments

Built for model tinkerers, curious minds, and anyone who enjoys watching artificial brains argue politely.

## License
This project is licensed under the Polyform Noncommercial License 1.0.0.
You may use, modify, and share the software for personal or research use.
Commercial use (products, paid services, monetization) is not permitted.
