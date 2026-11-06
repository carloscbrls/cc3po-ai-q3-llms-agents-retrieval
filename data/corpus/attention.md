# How Attention Works

Attention is the mechanism that lets a model weigh every token in the input
against every other token. Each token produces three vectors: a query, a key,
and a value. The query is compared against every key with a dot product, the
scores are scaled and passed through a softmax, and the resulting weights
average the values. Nothing more. The multi-head variant runs several of
these in parallel so different heads can specialise on different relations.
