# Why Transformers Replaced Recurrence

Recurrent networks process tokens one at a time, which limits parallelism
during training. Transformers removed that dependency: every token is
processed at once, and attention connects them. The trade-off is quadratic
cost in sequence length, which is why long-context work relies on sparse
attention variants, linear approximations, or careful engineering.
