#!/usr/bin/env python3
"""Verify data/recipe continuity, v3 invariants and transfer before freezing a new run."""
from __future__ import annotations

import argparse
import ast
import json
import math
import runpy
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_v3_common import ROOT, REFERENCE_RUN, DATA, MODELS, SETTINGS, model_path, now, require_run_root, sha256, transfer_report, write_json
from tools.preflight_ucra_v2 import dataset_audit

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def reference_audit(run, audit):
    import yaml
    state = read_json(REFERENCE_RUN/'state.json')
    if state['status'] != 'completed':
        raise RuntimeError('Completed A0/A4 reference required before freezing reference results')
    parent = read_json(REFERENCE_RUN/'train/contract.json')
    if any(parent['recipe'].get(k) != v for k,v in SETTINGS.items()):
        raise RuntimeError('Parent recipe does not match user parameters')
    if parent['early_stopping'] or 'patience' in parent['recipe']:
        raise RuntimeError('Parent early stopping protocol differs')
    if parent['dataset_yaml_sha256'] != sha256(DATA):
        raise RuntimeError('Dataset YAML changed from reference experiments')
    old_audit = read_json(REFERENCE_RUN/'train/dataset_audit.json')
    for split in ('train', 'val', 'test'):
        if audit['splits'][split]['content_sha256'] != old_audit['splits'][split]['content_sha256']:
            raise RuntimeError(f'Dataset content changed for {split}')
    # The registration files intentionally gain v3 names; all other original model,
    # data, loss, evaluator and training dependencies must retain their reference hash.
    exceptions = {'ultralytics/nn/tasks.py', 'ultralytics/nn/modules/__init__.py'}
    for rel, expected in parent['hashes'].items():
        if rel not in exceptions and sha256(ROOT/rel) != expected:
            raise RuntimeError(f'Reference dependency changed: {rel}')
    frozen_dir = run/'test/references'
    frozen_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for stage in ('A0', 'A4'):
        source = REFERENCE_RUN/'test'/f'{stage}_AllSeed_summary.json'
        summary = read_json(source)
        if len(summary['seeds']) != 3 or {r['seed'] for r in summary['seeds']} != {0,1,2}:
            raise RuntimeError(f'Incomplete reference: {stage}')
        for seed in range(3):
            train = REFERENCE_RUN/'train'/f'seed{seed}'/stage
            complete = read_json(train/'train_complete.json')
            if complete['epochs'] != 180 or complete['settings'] != SETTINGS or complete['early_stopping']:
                raise RuntimeError(f'Invalid parent training completion: {stage}/{seed}')
            if sha256(train/'weights/best.pt') != complete['weights_sha256']:
                raise RuntimeError('Parent weights changed')
            args = yaml.safe_load((train/'args.yaml').read_text(encoding='utf-8'))
            if any(args.get(key) != value for key,value in parent['recipe'].items()):
                raise RuntimeError(f'Effective parent training recipe differs: {stage}/{seed}')
            raw = read_json(REFERENCE_RUN/'test'/f'seed{seed}'/stage/'summary_metrics.json')
            scale = read_json(REFERENCE_RUN/'test'/f'seed{seed}'/stage/'scale_ap_metrics.json')['metrics']
            row = next(r for r in summary['seeds'] if r['seed'] == seed)
            for key, rawkey in (('P','precision'),('R','recall'),('mAP50','mAP50'),('mAP50_95','mAP50-95')):
                if not math.isclose(row[key],raw['metrics'][f'metrics/{rawkey}(B)'],rel_tol=0,abs_tol=1e-12):
                    raise RuntimeError('Reference summary disagrees with raw metrics')
            for key, rawkey in (('AP_S','APS'),('AP_M','APM'),('AP_L','APL')):
                if not math.isclose(row[key],scale[rawkey]/100,rel_tol=0,abs_tol=1e-12):
                    raise RuntimeError('Reference scale summary disagrees with raw metrics')
        dest = frozen_dir/source.name
        shutil.copy2(source,dest)
        records.append(dict(stage=stage, source=str(source), snapshot=str(dest), sha256=sha256(dest)))
    write_json(frozen_dir/'provenance.json', dict(reference_run=str(REFERENCE_RUN),state=state,
               sources=records, registration_only_exceptions=sorted(exceptions), verified_at=now()))
    write_json(run/'train/parent_contract.json',parent)
    return parent

def cuda_checks():
    import torch
    from ultralytics.nn.modules.ucra_v3 import UCRA1v3, UCRA2v3
    from ultralytics.utils.torch_utils import init_seeds
    assert torch.cuda.is_available()
    init_seeds(7,deterministic=True)
    reports=[]
    for kind, offset in ((UCRA1v3,None),(UCRA2v3,0.0),(UCRA2v3,0.35)):
        model = kind(64,32,**({'max_offset':offset} if offset is not None else {})).cuda()
        optim = torch.optim.SGD(model.parameters(),lr=0.2)
        gradients=[]
        for step in range(3):
            optim.zero_grad(set_to_none=True)
            deep=torch.randn(2,64,10,10,device='cuda',requires_grad=True)
            lateral=torch.randn(2,32,20,20,device='cuda',requires_grad=True)
            output=model([deep,lateral])
            if step == 0:
                assert torch.equal(output,torch.nn.functional.interpolate(deep,scale_factor=2,mode='nearest'))
            loss=(output-torch.randn_like(output)).square().mean()
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(deep.grad).all()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if kind is UCRA2v3:
                assert model.detail_out.weight.grad.abs().sum() > 0
                assert torch.isfinite(model.geom_gain_raw.grad).all()
                gradients.append(float(model.offset_head[-1].weight.grad.abs().sum()))
                if step == 0 and offset:
                    assert model.geom_gain_raw.grad.abs().sum() > 0
                if offset == 0:
                    parts=model.compute_components(deep,lateral)
                    assert torch.count_nonzero(parts['geometry_source']) == 0
            else:
                assert model.semantic_out.weight.grad.abs().sum() > 0
            optim.step()
        if offset:
            assert any(value > 0 for value in gradients[1:])
        reports.append(dict(module=kind.__name__,offset=offset,steps=3,offset_gradients=gradients))
    return reports

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args()
    run=require_run_root(args.run_root)
    if (run/'state.json').exists():
        raise FileExistsError('Existing chain cannot be re-preflighted')
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds,get_flops
    torch.set_num_threads(2)
    paths=list((ROOT/'ultralytics').rglob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    audit=dataset_audit(DATA)
    write_json(run/'train/dataset_audit.json',audit)
    parent=reference_audit(run,audit)
    print('REFERENCE_AND_DATA_AUDIT_PASS',flush=True)
    tests=runpy.run_path(str(ROOT/'tests/test_ucra_v3_csdr.py'))
    for name,function in tests.items():
        if name.startswith('test_'):
            function()
            print(name,'PASS',flush=True)
    runpy.run_path(str(ROOT/'tests/verify_ucra_v3_repo.py'),run_name='__main__')
    base_path=ROOT/'ultralytics/cfg/models/v13/yolov13.yaml'
    base_cfg=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    base_layers=base_cfg['backbone']+base_cfg['head']
    reports={}
    ref=None
    sample=torch.linspace(0,1,3*640*640).reshape(1,3,640,640)
    for stage,path in [('A0',base_path)]+[(s,model_path(s)) for s in MODELS]:
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        layers=cfg['backbone']+cfg['head']
        assert cfg['nc']==4 and len(layers)==33 and layers[-1][0]==[23,27,31]
        assert all(layers[i]==base_layers[i] for i in range(33) if i not in (15,19))
        init_seeds(0,deterministic=True)
        model=YOLO(str(path),verbose=False)
        transfer=transfer_report(model)
        model.model.eval()
        with torch.no_grad():
            prediction=model.model(sample)[0]
        assert torch.isfinite(prediction).all()
        if ref is None:
            ref=prediction.clone()
        else:
            assert torch.equal(ref,prediction),f'{stage}: full model zero-start differs'
        reports[stage]=dict(params=sum(p.numel() for p in model.model.parameters()),
            gflops=get_flops(model.model,imgsz=640),layers=33,forward_640=True,exact_baseline_start=True,transfer=transfer)
        print('MODEL',stage,reports[stage]['params'],reports[stage]['gflops'],'loaded',transfer['loaded_tensors'],flush=True)
        dest=run/'train/model_snapshots'/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        del model
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        cuda=cuda_checks()
    warning_messages=sorted(set(str(item.message) for item in captured))
    for message in warning_messages:
        print('CUDA_WARNING',message,flush=True)
    dependencies=['test.py','yolov13n.pt','ultralytics/cfg/default.yaml',
        'tools/ucra_v2_common.py','tools/preflight_ucra_v2.py','tools/run_ucra_v2.py',
        'tools/ucra_v3_common.py','tools/train_ucra_v3_worker.py','tools/run_ucra_v3.py','tools/preflight_ucra_v3.py',
        'scripts/run_ucra_v3.sh','scripts/wait_ucra_v3_after_ucra_v2.sh','experiments/urpc2020half_ucra_v3_ablation.yaml',
        'tests/test_ucra_v3_csdr.py','tests/verify_ucra_v3_repo.py']
    dependencies += [p.relative_to(ROOT).as_posix() for p in (ROOT/'ultralytics').rglob('*.py')]
    dependencies += [model_path(s).relative_to(ROOT).as_posix() for s in MODELS]
    frozen=dict(created_at=now(),data=str(DATA),dataset_yaml_sha256=sha256(DATA),recipe=parent['recipe'],
        parent_recipe=parent['recipe'],early_stopping=False,stages=list(MODELS),seeds=[0,1,2],phase2=False,
        reference_run=str(REFERENCE_RUN),initialization='YOLO(model_yaml).load(yolov13n.pt)',
        inherited_patience_ignored=40,hashes={rel:sha256(ROOT/rel) for rel in sorted(set(dependencies))},
        reference_hashes={p.relative_to(run).as_posix():sha256(p) for p in (run/'test/references').glob('*.json')})
    write_json(run/'train/contract.json',frozen)
    write_json(run/'train/preflight.json',dict(status='passed',at=now(),syntax_files=len(paths),
        models=reports,cuda_checks=cuda,cuda_warnings=warning_messages,reference_and_dataset_verified=True))
    print('PREFLIGHT_PASSED',str(run),flush=True)

if __name__=='__main__':
    main()
