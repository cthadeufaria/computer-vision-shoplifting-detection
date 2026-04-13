# Speckit Archive

Archived on: 2026-04-13

Purpose:
- Preserve all iOS-local Speckit artifacts before migrating the app workspace to OpenSpec.
- Leave the active `ios/` working tree clear of Speckit setup so a fresh OpenSpec initialization can be done manually.

Moved from the active `ios/` workspace:
- `.specify/`
- `specs/`
- `scripts/use-spec-kit-agent.sh`
- `scripts/generate_feature_coverage_matrix.py`
- `uml-class-diagram.md`
- `uml-class-diagram.html`
- `uml-sequence-diagram.md`
- `uml-sequence-diagram.html`
- `.claude/skills/speckit-analyze/`
- `.claude/skills/speckit-checklist/`
- `.claude/skills/speckit-clarify/`
- `.claude/skills/speckit-constitution/`
- `.claude/skills/speckit-implement/`
- `.claude/skills/speckit-plan/`
- `.claude/skills/speckit-specify/`
- `.claude/skills/speckit-tasks/`
- `.claude/skills/speckit-taskstoissues/`
- `.agents/skills/speckit-analyze/`
- `.agents/skills/speckit-checklist/`
- `.agents/skills/speckit-clarify/`
- `.agents/skills/speckit-constitution/`
- `.agents/skills/speckit-implement/`
- `.agents/skills/speckit-plan/`
- `.agents/skills/speckit-specify/`
- `.agents/skills/speckit-tasks/`
- `.agents/skills/speckit-taskstoissues/`

Archive root:
- `ios/archive/speckit-2026-04-13/`

Notes:
- No non-Speckit application source files were moved.
- Existing app code, Xcode project files, and non-Speckit skills were left in place.
