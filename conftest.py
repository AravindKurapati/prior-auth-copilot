"""Repo-root conftest: contain the single un-suppressible langgraph import warning
WITHOUT blanket-silencing warnings for the whole suite.

Background
----------
`langgraph.checkpoint.serde.jsonplus` builds a module-level `Reviver()` at import time.
`langchain_core.load.load.Reviver.__init__` emits a `LangChainPendingDeprecationWarning`
("The default value of `allowed_objects` will change...") on construction. That warning
therefore fires exactly once, during the first `import langgraph.*` in the session.

It cannot be neutralised with an ordinary `warnings.filterwarnings("ignore", ...)` /
`pyproject.toml` `filterwarnings` entry: `langchain_core.__init__` (and several langchain
sub-packages, re-triggered along the langgraph import chain) call
`surface_langchain_deprecation_warnings()`, which *prepends* `("default", LangChain*
DeprecationWarning)` filters to `warnings.filters`, jumping ahead of anything we put there
first. Whack-a-mole.

The reliable fix is to import langgraph HERE, once, inside a `catch_warnings(record=True)`
block that swallows whatever fires. By the time any test module imports langgraph it is
already in `sys.modules`, so nothing re-executes and nothing re-warns. Every *other*
warning is still governed by `filterwarnings = ["error", ...]` in pyproject.toml, so a
genuinely new warning anywhere in the suite fails it.
"""

import warnings

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    try:
        import langgraph.graph  # noqa: F401
        import langgraph.checkpoint.sqlite  # noqa: F401
    except ImportError:  # pragma: no cover - defensive; langgraph should import fine
        pass
