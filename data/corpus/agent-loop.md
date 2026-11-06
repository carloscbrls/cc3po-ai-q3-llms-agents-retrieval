# The Agent Loop

The loop reads like five lines and fails in fifty ways. ReAct interleaves
reasoning and acting. Plan-and-execute separates them. Scratchpad memory
keeps intermediate state. Every variant needs tripwires: a hard iteration
cap, a spend ceiling, and step logging, because an agent that appears stuck
is usually a loop that will never exit on its own.
