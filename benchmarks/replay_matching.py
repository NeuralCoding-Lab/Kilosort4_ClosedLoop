"""Compare both matchers on all 900 identical preprocessed input batches."""
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilosort import io, template_matching as tm
from kilosort.matching import TemplateMatcher


def timed(operation):
    torch.cuda.synchronize()
    start=time.perf_counter()
    value=operation()
    torch.cuda.synchronize()
    return value,time.perf_counter()-start


def main(root):
    torch.set_num_threads(16)
    saved=torch.load(root/'before/matching_input.pt',weights_only=False,map_location='cuda')
    ops,U=saved['ops'],saved['U']
    bfile=io.bfile_from_ops(ops,device=torch.device('cuda'))
    ctc=tm.prepare_matching(ops,U)
    matcher=TemplateMatcher(ops,U,ctc)
    iCC,_,iU,Ucc=tm.prepare_extract(ops['xc'],ops['yc'],U,ops['settings']['nearest_chans'],
                                  ops['settings']['position_limit'])
    wave=torch.arange(-(ops['nt']//2),ops['nt']//2+1,device='cuda')
    rows=[]
    total=0;total_after=0;common=0;maximum=0.;absolute_sum=0.;feature_values=0;nonidentical=0
    tied_batches=[];checked_batches=0;checked_spikes=0
    amp_max=0.;score_max=0.
    result=dict(batches=int(bfile.n_batches),templates=len(U),identical_times=True,
                identical_templates=True,identical_scores=True,identical_amplitudes=True,
                features_allclose=True,shared_preprocessed_input=True)
    started=time.perf_counter()
    for batch in range(bfile.n_batches):
        X=bfile.padded_batch_to_torch(batch,ops)
        fallback_before=matcher.statistics['tied_peak_fallbacks']
        order=['before','after'] if batch%2==0 else ['after','before']
        outputs={};durations={}
        for name in order:
            outputs[name],durations[name]=timed(lambda: tm.run_matching(ops,X,U,ctc)
                                                if name=='before' else matcher.match(X))
        st,amp,score,residual=outputs['before'];actual=outputs['after']
        fallback=matcher.statistics['tied_peak_fallbacks']>fallback_before
        same_times=torch.equal(st[:,0],actual.sample_indices)
        same_ids=torch.equal(st[:,1],actual.template_indices)
        result['identical_times'] &= same_times
        result['identical_templates'] &= same_ids
        result['identical_amplitudes'] &= torch.equal(amp[:,0],actual.amplitudes)
        result['identical_scores'] &= torch.equal(score[:,0],actual.scores)
        if fallback:
            # Both sides now execute original KS4, whose overlapping tied-peak
            # writes are themselves nondeterministic (see repeatability JSON).
            left_events=Counter(map(tuple,st.cpu().tolist()))
            right_events=Counter(zip(actual.sample_indices.cpu().tolist(),actual.template_indices.cpu().tolist()))
            common+=sum((left_events & right_events).values())
            tied_batches.append(dict(batch=int(batch),before_only=list((left_events-right_events).elements()),
                                     after_only=list((right_events-left_events).elements())))
        else:
            assert same_times,(batch,'non-tied spike times differ')
            assert same_ids,(batch,'non-tied templates differ')
            common+=len(st);checked_batches+=1;checked_spikes+=len(st)
            torch.testing.assert_close(amp[:,0],actual.amplitudes,rtol=3e-5,atol=3e-5)
            torch.testing.assert_close(score[:,0],actual.scores,rtol=3e-5,atol=3e-5)
            if len(st):
                amp_max=max(amp_max,float((amp[:,0]-actual.amplitudes).abs().max()))
                score_max=max(score_max,float((score[:,0]-actual.scores).abs().max()))
            indices=(iCC[:,iU[st[:,1:2]]],st[:,:1]+wave)
            left=residual[indices] @ ops['wPCA'].T + amp*Ucc[:,st[:,1]]
            right=actual.residual[indices] @ ops['wPCA'].T + actual.amplitudes[:,None]*Ucc[:,st[:,1]]
            torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-5)
            delta=(left-right).abs()
            if delta.numel():
                maximum=max(maximum,float(delta.max()))
                absolute_sum+=float(delta.double().sum())
                nonidentical+=int(torch.count_nonzero(delta))
                feature_values+=delta.numel()
        total+=len(st)
        total_after+=len(actual.sample_indices)
        rows.append(dict(batch=int(batch),spikes=len(st),spikes_after=len(actual.sample_indices),
                         legacy_tied_fallback=fallback,identical_times=same_times,identical_templates=same_ids,**durations))
        if batch%50==0 or batch==bfile.n_batches-1:
            print(f'Validated {batch+1}/{bfile.n_batches}; {total:,} spikes; max tF error {maximum:.3g}',flush=True)
            (root/'full_matching_batches.json').write_text(json.dumps(rows,indent=2))
    result.update(spikes_before=total,spikes_after=total_after,common_spikes_exact=common,
        exact_event_f1=2*common/(total+total_after),legacy_tied_batches=tied_batches,
        strict_non_tied_batches=checked_batches,strict_non_tied_spikes=checked_spikes,
        identical_spike_table=bool(result['identical_times'] and result['identical_templates'] and result['identical_scores']),
        feature_max_abs=maximum,feature_mean_abs=absolute_sum/max(1,feature_values),
        feature_values=feature_values,feature_nonidentical_values=nonidentical,
        amplitude_max_abs=amp_max,score_max_abs=score_max,
        before_seconds=sum(r['before'] for r in rows),after_seconds=sum(r['after'] for r in rows),
        wall_seconds=time.perf_counter()-started,matching_stats=matcher.statistics.copy())
    result['speedup']=result['before_seconds']/result['after_seconds']
    (root/'full_matching_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    matcher.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
