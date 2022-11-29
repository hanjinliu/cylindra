from typing import Sequence

import pytest
from cylindra._list import DataFrameList
import pandas as pd
from pandas.testing import assert_frame_equal

def _assert_eq(a: Sequence[pd.Series], b: Sequence[pd.Series]):
    assert len(a) == len(b)
    for a0, b0 in zip(a, b):
        assert a0.equals(b0)
    
def test_agg_pos():
    dfl = DataFrameList([
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        pd.DataFrame({"a": [5, 4, 3], "b": [2, 1, 2]}),
        pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]}),
    ])
    
    _sum = dfl.agg_pos("sum")
    _assert_eq(_sum, [pd.Series({"a": 6, "b": 15}), pd.Series({"a": 12, "b": 5}), pd.Series({"a": 3, "b": 3})])
    
    _min = dfl.agg_pos("min")
    _assert_eq(_min, [pd.Series({"a": 1, "b": 4}), pd.Series({"a": 3, "b": 1}), pd.Series({"a": 1, "b": 1})])
    
    _max = dfl.agg_pos("max")
    _assert_eq(_max, [pd.Series({"a": 3, "b": 6}), pd.Series({"a": 5, "b": 2}), pd.Series({"a": 1, "b": 1})])

def test_agg_id():
    dfl = DataFrameList([
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        pd.DataFrame({"a": [5, 4], "b": [2, 1]}),
        pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]}),
    ])
    
    assert_frame_equal(dfl.agg_id("sum", align="top"), pd.DataFrame({"a": [7, 7, 4], "b": [7, 7, 7]}))
    assert_frame_equal(dfl.agg_id("size", align="top"), pd.DataFrame({"a": [3, 3, 2], "b": [3, 3, 2]}))
    
    assert_frame_equal(dfl.agg_id("sum", align="bottom"), pd.DataFrame({"a": [2, 8, 8], "b": [5, 8, 8]}))
    assert_frame_equal(dfl.agg_id("size", align="bottom"), pd.DataFrame({"a": [2, 3, 3], "b": [2, 3, 3]}))

    with pytest.raises(ValueError):
        dfl.agg_id("sum", align="none")
    
    # test just works
    dfl.agg_id("mean")
    dfl.agg_id("median")
    dfl.agg_id("std")
    dfl.agg_id("sem")

def test_collect():
    dfl = DataFrameList([
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        pd.DataFrame({"a": [5, 4, 3], "b": [2, 1, 2]}),
    ])
    
    out = dfl.collect()
    assert_frame_equal(out, pd.DataFrame({"a": [1, 2, 3, 5, 4, 3], "b": [4, 5, 6, 2, 1, 2]}))

def test_indexer():
    dfl = DataFrameList([
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        pd.DataFrame({"a": [5, 4, 3], "b": [2, 1, 2]}),
    ])
    
    out = dfl.iloc[1:]
    assert_frame_equal(out[0], pd.DataFrame({"a": [2, 3], "b": [5, 6]}))
    assert_frame_equal(out[1], pd.DataFrame({"a": [4, 3], "b": [1, 2]}))
    