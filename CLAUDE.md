# Project Memory / Research Retrieval

`BANK.md` is the historical source of truth but should NOT be read in full during normal work.

When historical project context is needed:

1. Check `research_memory/CURRENT.md` for the current state.
2. Use `research_memory/INDEX.md` to understand where relevant information is located.
3. Use `research_memory/tools/search_memory.py` to retrieve relevant historical entries.
4. Read only the returned relevant entries/sections.
5. Only read `BANK.md` directly when absolutely necessary and when targeted retrieval cannot provide the required information.

Useful commands:
- `python research_memory/tools/search_memory.py "QUERY"`
- `python research_memory/tools/search_memory.py "QUERY" --recent N`
- `python research_memory/tools/search_memory.py --entry ENTRY_NUMBER`

Do not ask the user to manually provide or upload `BANK.md` when the repository is available.

**Important behavior:**
Do NOT search the memory system for every trivial coding task. Use retrieval only when the task depends on:
- previous experiments
- model versions
- dataset versions
- training results
- evaluation results
- previous technical decisions
- previous failures
- project history
- why a particular approach was chosen

For ordinary source-code tasks where the relevant code itself is sufficient, do not unnecessarily search the historical memory.
