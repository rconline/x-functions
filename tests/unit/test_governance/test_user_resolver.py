from spark_ai_functions.governance.user_resolver import (
    ChainedUserResolver,
    ExplicitUserResolver,
    SparkPropertyUserResolver,
    UNKNOWN_USER,
    default_chain,
)


class _FakeTC:
    def __init__(self, props):
        self._p = props

    def getLocalProperty(self, key):
        return self._p.get(key)


def test_explicit_resolver():
    assert ExplicitUserResolver("alice").resolve(None) == "alice"


def test_explicit_resolver_empty_treated_as_unknown():
    assert ExplicitUserResolver("").resolve(None) == UNKNOWN_USER


def test_spark_property_reads_key():
    tc = _FakeTC({"spark.ai.user": "bob@co"})
    assert SparkPropertyUserResolver().resolve(tc) == "bob@co"


def test_spark_property_missing_returns_unknown():
    assert SparkPropertyUserResolver().resolve(_FakeTC({})) == UNKNOWN_USER


def test_chained_skips_unknown_and_returns_first_hit():
    chain = ChainedUserResolver(
        SparkPropertyUserResolver(),        # returns UNKNOWN_USER for empty tc
        ExplicitUserResolver("fallback"),
    )
    assert chain.resolve(_FakeTC({})) == "fallback"


def test_default_chain_uses_explicit_when_nothing_else():
    chain = default_chain(explicit_user="dev@localhost")
    assert chain.resolve(_FakeTC({})) == "dev@localhost"


def test_default_chain_without_explicit_returns_unknown_in_isolation():
    chain = default_chain(explicit_user=None)
    assert chain.resolve(_FakeTC({})) == UNKNOWN_USER


def test_chained_requires_at_least_one():
    import pytest
    with pytest.raises(ValueError):
        ChainedUserResolver()
