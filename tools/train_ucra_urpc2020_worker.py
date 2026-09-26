#!/usr/bin/env python3
"""One fresh full-URPC2020 training/test worker with audited transfer weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_urpc2020_common import ROOT, SETTINGS, MODELS, model_path, now, require_run_root, sha256, transfer_report, write_json, verify_contract


def no_early_stop(trainer):
    # Remove patience from the submitted recipe and disable the inherited default stopper.
    trainer.stopper.patience = math.inf
    trainer.stopper.possible_stop = False
    print('EARLY_STOPPING_DISABLED: all 180 epochs required', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--stage', required=True, choices=MODELS)
    parser.add_argument('--seed', required=True, type=int, choices=(0, 1, 2))
    parser.add_argument('--mode', required=True, choices=('train', 'test'))
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    frozen = verify_contract(run)
    output = run / args.mode / f'seed{args.seed}' / args.stage
    train_output = run / 'train' / f'seed{args.seed}' / args.stage
    if output.exists():
        raise FileExistsError(f'Refusing duplicate output: {output}')
    print(f'LOCAL_ULTRALYTICS_ROOT={ROOT}; stage={args.stage}; seed={args.seed}; mode={args.mode}; data={frozen["data"]}', flush=True)
    if args.mode == 'test':
        import test as evaluator
        sys.argv = [str(ROOT / 'test.py'), '--weights', str(train_output / 'weights/best.pt'),
                    '--data', frozen['data'], '--project', str(output.parent), '--name', output.name,
                    '--device', '0', '--batch', '16', '--imgsz', '640', '--workers', '2', '--no-plots']
        evaluator.main()
        for name in ('summary_metrics.json', 'scale_ap_metrics.json'):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(args.seed, deterministic=True)
    model = YOLO(str(model_path(args.stage)))
    write_json(output.parent / f'{args.stage}.pretrained_load.json', transfer_report(model))
    model.add_callback('on_train_start', no_early_stop)

    def record_best_epoch(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(output / 'best_epoch.json', dict(epoch=trainer.epoch + 1, fitness=float(trainer.fitness)))

    model.add_callback('on_model_save', record_best_epoch)
    options = dict(frozen['recipe'])
    options.update(SETTINGS, data=frozen['data'], seed=args.seed, project=str(output.parent),
                   name=output.name, exist_ok=False, resume=False)
    if 'patience' in options:
        raise RuntimeError('patience must not be supplied')
    started = perf_counter()
    model.train(**options)
    if model.trainer.epoch + 1 != SETTINGS['epochs']:
        raise RuntimeError(f'Incomplete training: {model.trainer.epoch + 1} epochs')
    best = output / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    with (output / 'results.csv').open(encoding='utf-8') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != SETTINGS['epochs'] or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise RuntimeError('Invalid training results')
    best_epoch = json.loads((output / 'best_epoch.json').read_text(encoding='utf-8'))['epoch']
    write_json(output / 'train_complete.json', dict(stage=args.stage, seed=args.seed, data=frozen['data'],
        epochs=len(rows), best_epoch=best_epoch, settings=SETTINGS, early_stopping=False,
        seconds=perf_counter()-started, completed_at=now(), weights_sha256=sha256(best)))


if __name__ == '__main__':
    main()
