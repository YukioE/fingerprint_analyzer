"""Detect whether any input signals opt-in Replace-By-Fee."""

from .base import detector


@detector("default_opt_in_rbf_set")
def detect(ctx):
    """Return the replace-by-fee signal precomputed in the context."""
    return ctx.rbf
