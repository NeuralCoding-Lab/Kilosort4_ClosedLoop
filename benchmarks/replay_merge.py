"""Replay a frozen full-recording merge and record every CCG decision."""
import argparse
import functools
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from benchmark_ks4 import Recorder


def digest(value):
    array=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    checksum=hashlib.sha256()
    for start in range(0,len(array),100000):
        checksum.update(np.ascontiguousarray(array[start:start+100000]).view(np.uint8))
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=checksum.hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    sys.path.insert(0,str(args.source.resolve()))
    from kilosort import CCG,template_matching as tm
    torch.set_num_threads(16)
    saved=torch.load(args.checkpoint,weights_only=False)
    ops,Wall,clu,st,features=(saved[k] for k in ('ops','Wall','clu','st','tF'))
    decisions=[]
    original_check=CCG.check_CCG
    @functools.wraps(original_check)
    def traced(*values,**kwargs):
        frame=sys._getframe(1)
        result=original_check(*values,**kwargs)
        local=frame.f_locals
        decisions.append(dict(caller=frame.f_code.co_name,seed=int(local.get('kk',-1)),
            candidate=int(local.get('jj',-1)),left_spikes=len(values[0]),
            right_spikes=len(values[1]) if len(values)>1 and values[1] is not None else len(values[0]),
            refractory=bool(result[0]),cross_refractory=bool(result[1]),rate=float(result[2])))
        if len(decisions)%200==0:
            print(f"CCG decisions recorded: {len(decisions)}; caller={frame.f_code.co_name}; seed={local.get('kk',-1)}",flush=True)
        return result
    CCG.check_CCG=traced
    recorder=Recorder(args.output)
    recorder.wrap(tm,'merging_function','final_merge')
    recorder.thread.start()
    result={}
    try:
        output=tm.merging_function(ops,Wall,clu,st,features,**saved['kwargs'])
        recorder.stage='output_hashing'
        keys=('Wall','clu','is_ref','st','tF')
        result=dict(success=True,input_checkpoint=str(args.checkpoint.resolve()),
            source=str(args.source.resolve()),outputs={k:digest(v) for k,v in zip(keys,output)},
            ccg_calls=len(decisions),accepted_pairs=sum(d['cross_refractory'] for d in decisions if d['caller']=='merging_function'))
        (args.output/'decisions.json').write_text(json.dumps(decisions,indent=2))
        (args.output/'source_hashes.json').write_text(json.dumps({str(p.relative_to(args.source)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source/'kilosort').rglob('*.py')},indent=2))
    finally:
        recorder.stop.set();recorder.thread.join()
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
