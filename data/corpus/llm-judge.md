# LLM as Judge

An LLM judge scores outputs against a rubric. It is cheaper and faster than
human review, and it is consistent enough to catch large regressions. It is
also biased toward longer answers and toward its own family of models, so
it should be calibrated against human labels before it is trusted.
