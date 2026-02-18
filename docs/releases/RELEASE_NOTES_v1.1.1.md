# PolyCouncil v1.1.1 Release Notes

**Release Date:** February 18, 2026

## Highlights

- Added first-class multi-provider support:
  - LM Studio
  - Ollama
  - OpenAI-compatible hosted APIs
- Added hosted API presets:
  - OpenAI
  - OpenRouter
  - Google Gemini (OpenAI-compatible endpoint)
- Added mixed-provider councils: local and hosted models can now run in the same round.

## UX and Workflow Improvements

- Provider setup is now centered in the main screen for faster use.
- Added additive model loading workflow:
  - **Add Models** appends provider models to the current list.
  - **Replace List** explicitly rebuilds the list from one provider.
  - **Clear Model List** removes all loaded models.
- Added saved provider profiles:
  - **Add Provider**, then **Use** or **Add Models** per saved provider.
- Single-voter and selection workflows now work with provider-tagged model names.

## Stability and Reliability Fixes

- Fixed incorrect LM Studio model endpoint usage (`/v1/models` enforced for LM Studio).
- Hardened provider switching and model list merge paths.
- Added hosted-provider key validation before model discovery.
- Added crash logging (`polycouncil_crash.log`) for unhandled exceptions.
- Fixed discussion mode argument mismatch and added cancellation handling.
- Fixed discussion export question capture.
- Reduced capability detection overhead with metadata caching.
- Replaced mutable default list arguments in runtime paths.

## Developer and QA

- Added automated tests for:
  - Provider config/default behavior and endpoint construction
  - Capability detection helpers
  - Mixed-provider vote routing
  - Voting JSON parsing/validation
  - Discussion manager provider/cancellation behavior
