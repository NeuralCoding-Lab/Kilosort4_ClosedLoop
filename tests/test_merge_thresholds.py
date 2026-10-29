"""Golden merge outputs at thresholds between adjacent float32 values."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from kilosort import CCG, template_matching as tm


@pytest.mark.parametrize('ulp_offset', [-.25, .25, .75])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_cpu_threshold_preserves_original_tensor_scalar_cast(monkeypatch, ulp_offset, device):
    source=Path(__file__).resolve().parents[1]/'benchmarks/ks4_cpu_merge_20261027/baseline/kilosort/template_matching.py'
    spec=importlib.util.spec_from_file_location('merge_before_cpu_scores',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    W=torch.eye(7,device=device)
    Wall=torch.zeros(2,3,7,device=device)
    Wall[:,0,3]=1.
    Wall[1,1,3]=.5
    labels=np.tile(np.arange(2),12).astype('int32')
    st=np.column_stack((np.arange(24)*100,labels,np.ones(24))).astype('float64')
    features=torch.zeros(24,3,7)
    ops=dict(nt=7,wPCA=W,fs=30000,settings=dict(acg_threshold=.2,ccg_threshold=.25))
    normalized=Wall/(1e-6+Wall.square().sum((1,2),keepdim=True).sqrt())
    lag=torch.nn.functional.conv1d(W[:,None],W[:,None],padding=7).flip(2)
    spatial=torch.einsum('lk,jlm->jkm',normalized[1],normalized)
    score=torch.einsum('jkm,kml->jl',spatial,lag).max(1).values[0].item()
    ulp=float(np.nextafter(np.float32(score),np.float32(np.inf))-np.float32(score))
    threshold=score+ulp_offset*ulp
    monkeypatch.setattr(CCG,'refract',lambda *a,**kw:(np.ones(2),np.zeros(2)))
    monkeypatch.setattr(CCG,'check_CCG',lambda *a,**kw:(True,True,0.))
    expected=original.merging_function(ops,Wall.clone(),labels.copy(),st.copy(),features.clone(),
        r_thresh=threshold,check_dt=False,device=torch.device(device))
    actual=tm.merging_function(ops,Wall.clone(),labels.copy(),st.copy(),features.clone(),
        r_thresh=threshold,check_dt=False,device=torch.device(device))
    for a,b in zip(actual,expected):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:np.testing.assert_array_equal(a,b)
