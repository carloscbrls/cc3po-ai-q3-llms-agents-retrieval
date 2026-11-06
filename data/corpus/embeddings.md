# Embeddings and Vector Space

An embedding maps text into a fixed-length vector so that semantic
similarity becomes geometric proximity. Two sentences about refunds that use
different words can land close together. The quality of a retrieval system
is bounded by the quality of the embedding model, which is why swapping in a
better embedder can beat switching to a much larger language model.
