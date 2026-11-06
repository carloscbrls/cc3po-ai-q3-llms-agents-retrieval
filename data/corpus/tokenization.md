# How Text Becomes Tokens

Tokenizers split raw text into subword units using algorithms like byte pair
encoding. Common words become single tokens, rare words fragment into
pieces. This is why cost scales with tokens rather than words, why unusual
spellings and emoji consume more budget, and why a model may split a
customer's name across four or five tokens.
