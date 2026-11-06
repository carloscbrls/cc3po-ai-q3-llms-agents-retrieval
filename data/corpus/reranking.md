# Reranking Retrieved Candidates

A first-stage retriever casts a wide net and returns candidates quickly. A
reranker then scores those candidates more carefully with a slower, more
accurate model. Narrowing fifty candidates down to five with a cross-encoder
often changes the final answer more than switching to a ten times larger
generator.
