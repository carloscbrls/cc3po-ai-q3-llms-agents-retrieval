# Tools and Function Calling

Function calling lets a model emit structured requests to your code. You
define the tool with a JSON schema, the model proposes a call, your code
executes it and returns a result. The model never runs your code itself.
Design for narrow tools, explicit descriptions, typed returns, and
deterministic fallbacks.
