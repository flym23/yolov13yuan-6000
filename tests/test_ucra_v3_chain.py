"""Exercise chain cleanup and acceptance decisions without training any models."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('fcntl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_ucra_v3 as chain

def test_worker_failure_stops_and_reaps_siblings(tmp_path, monkeypatch):
    run=tmp_path/'run'
    chain.write_json(run/'train/preflight.json',{'status':'passed'})
    parent=tmp_path/'parent_state.json'
    chain.write_json(parent,{'status':'completed'})
    for stage in ('A0','A4'):
        chain.write_json(run/'test/references'/f'{stage}_AllSeed_summary.json',{'seeds':[],'statistics':{}})
    real_popen=subprocess.Popen
    children=[]
    def launch(command,**kwargs):
        seed=int(command[command.index('--seed')+1])
        script='import sys;sys.exit(7)' if seed==0 else 'import time;time.sleep(120)'
        process=real_popen([sys.executable,'-c',script],**kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(chain,'require_run_root',lambda _:run)
    monkeypatch.setattr(chain,'verify_contract',lambda _: {})
    monkeypatch.setattr(chain.subprocess,'Popen',launch)
    monkeypatch.setattr(sys,'argv',['chain','--run-root',str(run),'--upstream-state',str(parent)])
    with pytest.raises(RuntimeError,match='exit 7'):
        chain.main()
    assert len(children)==3 and all(p.poll() is not None for p in children)
    state=chain.load(run/'state.json')
    assert state['status']=='failed' and state['worker_pids']=={}

def test_decision_keeps_context_winner_and_rejects_single_seed_spike(tmp_path):
    rows=[]
    base=dict(P=.8,R=.75,mAP50=.8,mAP50_95=.46,AP_S=.14,AP_M=.38,AP_L=.49)
    for stage in ('A0','A4','B1','B2','B3'):
        for seed in range(3):
            row=dict(base,stage=stage,seed=seed)
            if stage=='B1':
                row.update(mAP50_95=.465,AP_S=.145)
            if stage=='B2':
                row.update(mAP50_95=.462,AP_S=.142)
            if stage=='B3':
                row.update(mAP50_95=.47,AP_S=.17 if seed==0 else .139)
            rows.append(row)
    (tmp_path/'test').mkdir()
    chain.finalize(tmp_path,rows,[])
    result=chain.load(tmp_path/'test/decision.json')
    assert result['acceptance']['B1']['accepted']
    assert result['acceptance']['B2']['accepted']
    assert not result['acceptance']['B3']['accepted']
    assert result['preferred_structure']=='B1'
    assert not result['phase2_executed']
