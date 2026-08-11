"""R5: the shared canonicalisation module reproduces every stored gold path.

Each gold_tool_path_display is replayed through the fixed tools module and
canonicalised with cx_agent_bench.canonical; the result must equal the stored
gold_tool_path_json byte for byte. This pins builder and harness to one normal
form: if either side drifted, this test breaks.
"""

import pandas as pd
import pytest

from cx_agent_bench.canonical import canonical_json
from cx_agent_bench.harness import load_tools_module
from cx_agent_bench.tasks_gold import TASKS_GOLD_CSV
from cx_agent_bench.tasks_public import load_dataset

GOLD = pd.read_csv(TASKS_GOLD_CSV, index_col=0)
DATA = {name: load_dataset(name) for name in ("easy", "hard")}
TOOLS_MODULE = load_tools_module()


@pytest.mark.parametrize("row", [r for _, r in GOLD.iterrows()],
                         ids=GOLD.task_id.tolist())
def test_replayed_gold_path_canonicalises_to_stored_json(row):
    tools = TOOLS_MODULE.Tools(DATA[row.dataset])
    namespace = {name: getattr(tools, name)
                 for name in ("filter", "summarise", "rank", "ztest", "answer")}
    for call in eval(row.gold_tool_path_display):
        eval(call, {"__builtins__": {}}, namespace)
    assert canonical_json(tools) == row.gold_tool_path_json


def test_handle_names_do_not_matter():
    """An agent whose handles are numbered differently must canonicalise the same."""
    a = TOOLS_MODULE.Tools(DATA["easy"])
    a.filter(industry="Banking", aspect="app-website")
    a.summarise("s1")

    b = TOOLS_MODULE.Tools(DATA["easy"])
    b.filter(industry="Fashion")            # burns s1 on something else
    b.filter(industry="Banking", aspect=["app-website"])
    b.summarise("s2")
    assert (canonical_json(a.calls) ==
            canonical_json(b.calls[1:]))    # same work, different handle names
