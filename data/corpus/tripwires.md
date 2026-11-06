# Guardrails for Long Running Agents

A tripwire is a limit that stops a runaway agent. A maximum iteration count
prevents infinite loops. A cost ceiling prevents a silent money leak. An
approval gate pauses before irreversible actions. A run log records every
tool call so failures can be diagnosed. None of this makes the agent
smarter; it makes the failure survivable.
