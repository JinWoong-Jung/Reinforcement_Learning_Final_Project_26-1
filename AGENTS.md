# Agent Instructions

Use this file as the single top-level entry point for repository guidance.

Before exploring the repository, read these files in order:

1. `.agents/README.md`
2. `.agents/AGENTS.md`
3. `.agents/rules/README.md`
4. The task-specific rule file you actually need

Default behavior for this repository:

- Use the `.agents` documents as the primary context.
- Do not start by scanning the entire codebase.
- Only inspect files that are relevant to the task after reading the `.agents` docs.
- If documentation and implementation differ, trust the code and then update the `.agents` docs when appropriate.

Reference map:

- `.agents` index: `.agents/README.md`
- Project overview: `.agents/AGENTS.md`
- Rules index: `.agents/rules/README.md`
- RL agent behavior and training scope: `.agents/rules/agents.md`
- Environment, state, action, reward, and dynamics: `.agents/rules/env.md`
