"""Preserve reconstruction, edge filling and projection for signed shifts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from kilosort.template_matching import roll_features


@pytest.mark.parametrize('dt',[-3,0,2])
@pytest.mark.parametrize('kind',['numpy_bool','numpy_int','torch_bool','torch_int'])
def test_native_feature_rows_equal_original_indexing(dt,kind):
    source=Path(__file__).resolve().parents[1]/'benchmarks/ks4_cpu_merge_20261027/baseline/kilosort/template_matching.py'
    spec=importlib.util.spec_from_file_location('original_roll_features',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    torch.manual_seed(83)
    W=torch.linalg.qr(torch.randn(9,4)).Q.T.contiguous()
    features=torch.randn(101,6,4)
    Wall=torch.randn(3,6,4)
    selected=np.array([0,9,31,68,100])
    mask=np.isin(np.arange(len(features)),selected)
    index={'numpy_bool':mask,'numpy_int':selected,'torch_bool':torch.from_numpy(mask),
           'torch_int':torch.from_numpy(selected)}[kind]
    expected=original.roll_features(W,features.clone(),Wall.clone(),index,1,dt)
    actual=roll_features(W,features.clone(),Wall.clone(),index,1,dt)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Every unselected feature row must be preserved exactly.
    torch.testing.assert_close(actual[0][torch.from_numpy(~mask)],features[torch.from_numpy(~mask)],rtol=0,atol=0)
