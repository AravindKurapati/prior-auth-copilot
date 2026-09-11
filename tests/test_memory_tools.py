from pa_copilot.memory.tools import build_memory_tools

NS = ("pa", "member", "M100001")

# NOTE: PolicyStore (Task 4) subclasses langgraph's *sync* SqliteStore, which
# raises NotImplementedError from its `abatch` ("Consider using AsyncSqliteStore
# instead."). LangMem's manage/search tools route their async path through
# `store.aput`/`store.asearch`, which in turn call `abatch` -- so `.ainvoke(...)`
# is unusable against this store today. Per the task brief's guidance, only the
# sync `.invoke(...)` path is exercised here; the load-bearing assertion is the
# policy-governance one below.


def test_manage_then_search_roundtrip(memory_store):
    manage, search = build_memory_tools(memory_store, NS)
    manage.invoke({"content": "Member had lumbar MRI approved in 2025."})
    hits = search.invoke({"query": "lumbar MRI history"})
    assert "lumbar" in str(hits).lower()


def test_manage_write_is_policy_governed(memory_store):
    manage, _ = build_memory_tools(memory_store, NS)
    manage.invoke({"content": "note governed by PolicyStore"})
    items = memory_store.search(NS, limit=10)
    assert items and all("importance" in i.value for i in items)
