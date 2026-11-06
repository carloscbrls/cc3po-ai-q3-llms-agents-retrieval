# Retrieval Augmented Generation

Retrieval augmented generation grounds a language model in retrieved
documents. The pipeline is ingest, chunk, embed, index, retrieve, rerank,
assemble, generate, and cite. The model does not know your data, and
fine-tuning is the wrong tool for fresh or regulated facts, so retrieval
supplies the facts at inference time instead.
