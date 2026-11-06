# Prompt Injection

A prompt injection is hostile text that tries to make the model ignore its
instructions. A tool-using agent that reads a webpage can encounter an
instruction to exfiltrate data and may follow it. Treat all retrieved
content as untrusted input, give tools narrow scopes, and never let a model
action bypass a permission check.
