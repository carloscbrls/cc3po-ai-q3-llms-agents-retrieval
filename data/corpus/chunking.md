# Chunking Strategies

Chunking decides what a retrievable unit looks like. Too small and context
is lost. Too large and the embedding averages unrelated topics, so retrieval
gets worse. Chunk size should be tuned per corpus, and chunks should carry
overlap so a sentence spanning a boundary is still retrievable.
