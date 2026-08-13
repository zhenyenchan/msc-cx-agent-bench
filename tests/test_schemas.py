"""R2: the field vocabularies documented in the tool schemas must match the data.

The labels are part of the agent-facing documentation (no agent can name a slice
it cannot see), so a vocabulary drifting from the corpus would silently change
task difficulty.
"""

import pytest

from cx_agent_bench.tasks_public import load_dataset
from cx_agent_bench.tool_schemas import ASPECTS, INDUSTRIES, ORGS, SENTIMENTS


@pytest.mark.parametrize("dataset", ["easy", "hard"])
def test_schema_vocabularies_match_dataset(dataset):
    df = load_dataset(dataset)
    assert set(df["industry"].unique()) == set(INDUSTRIES)
    assert set(df["org"].unique()) == set(ORGS)
    assert set(df["aspect"].unique()) == set(ASPECTS)
    assert set(df["sentiment"].unique()) == set(SENTIMENTS)
