# What Is an Agent

An agent is a language model plus tools plus a loop plus state. The model
decides which tool to call, the loop runs it and returns the result, and the
model decides what to do next. The loop is where all the production problems
live: infinite loops, tool thrash, context exhaustion, and premature claims
of success.
