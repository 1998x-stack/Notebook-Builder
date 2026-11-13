"""Read-only, fail-closed evaluation of paper-reproduction EVIDENCE, not a claim of independent audit.

Never conflate an executed synthetic tutorial, a real-dataset partial run, a
completed experiment, or all experiment families for an entire paper.

The JSON run manifests are reported by the executing experiment. This validator
checks schemas, cross-field consistency and declared benchmark coverage; it
cannot independently establish whether unseen data/metrics were fabricated.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any

SCOPES = {
    'cifar10_single': None,  # One completed CIFAR experiment; NOT the full paper
    'cifar10_paper': {('plain', d) for d in (20, 32, 44, 56)} |
                     {('resnet', d) for d in (20, 32, 44, 56, 110, 1202)},
    'imagenet_paper': {('plain', d) for d in (18, 34)} |
                       {('resnet', d) for d in (18, 34, 50, 101, 152)},
    'detection_paper': {('faster_rcnn_vgg16', 'voc'), ('faster_rcnn_resnet101', 'voc'),
                        ('faster_rcnn_vgg16', 'coco'), ('faster_rcnn_resnet101', 'coco')},
}
DATASETS = {
    'cifar10': dict(name='CIFAR10', classes=10, test=10000, train=45000, train_final=50000),
    'imagenet': dict(name='ImageNet-1k/ILSVRC2012', classes=1000, test=50000, train=1280000),
}


def _get_family(run: dict) -> str:
    return str(run.get('family', '')).lower()


def _valid_run(run: Any, scope: str) -> list[str]:
    problems = []
    if not isinstance(run, dict):
        return ['Experiment is not an object']
    def err(s): problems.append(s)
    data = run.get('dataset') or {}
    if not isinstance(data, dict): return ['dataset is not an object']
    if data.get('origin') != 'real': err('dataset.origin must equal real')
    if not data.get('checksum_verified'): err('missing positive dataset checksum verification')
    if not isinstance(data.get('fingerprint_sha256'), str) or len(data['fingerprint_sha256']) != 64:
        err('missing 64-character dataset fingerprint')
    if not run.get('paper_profile'): err('missing paper_profile')
    if run.get('skipped'): err('run is skipped')
    if not run.get('checkpoint_sha256') or len(str(run.get('checkpoint_sha256'))) != 64:
        err('missing checkpoint SHA256')
    if not isinstance(run.get('seed'), int): err('seed missing')
    if not isinstance(run.get('accuracy_top1'), (int, float)) or not 0 <= run['accuracy_top1'] <= 1:
        err('missing accuracy_top1 in [0, 1]')
    done, wanted = run.get('steps_completed'), run.get('steps_required')
    if not isinstance(done, int) or not isinstance(wanted, int) or wanted < 1 or done != wanted:
        err('steps_completed must equal positive steps_required')
    family = _get_family(run)
    if family == 'cifar10':
        if run.get('paper_profile') != 'he2015_cifar10_64k_sgd': err('CIFAR protocol tag mismatch')
        if wanted != 64000: err('CIFAR-10 paper schedule requires 64000 optimizer steps')
        if data.get('name') != 'CIFAR10' or data.get('classes') != 10: err('wrong CIFAR10 dataset/classes')
        if data.get('train_count') != 50000: err('CIFAR final paper benchmark requires all 50000 training images')
        if data.get('test_count') != 10000 or run.get('test_samples_evaluated') != 10000:
            err('full 10000-example CIFAR test evaluation required')
        if run.get('shortcut') not in ('A', 'none'): err('CIFAR shortcut must be paper Option A or plain')
        if run.get('batch_size') != 128: err('CIFAR paper batch size 128 required')
        if run.get('optimizer') != 'SGD(momentum=0.9,weight_decay=1e-4)': err('CIFAR SGD protocol mismatch')
        if run.get('scheduler') != 'step_32000_48000': err('CIFAR LR milestones mismatch')
        if run.get('augmentation') != 'pad4_crop32_hflip': err('CIFAR augmentation mismatch')
        if (run.get('kind') == 'plain') != (run.get('shortcut') == 'none'): err('plain/resnet shortcut contradiction')
        if run.get('depth') not in (20,32,44,56,110,1202): err('unsupported CIFAR paper depth')
    elif family == 'imagenet':
        if run.get('paper_profile') != 'he2015_imagenet_600k_sgd': err('ImageNet protocol tag mismatch')
        if wanted != 600000: err('ImageNet paper requires up to 600k optimizer steps')
        if data.get('name') != 'ImageNet-1k/ILSVRC2012' or data.get('classes') != 1000:
            err('wrong ImageNet dataset/classes')
        if data.get('train_count',0) < 1200000: err('ImageNet full training set required')
        if data.get('test_count') != 50000 or run.get('test_samples_evaluated') != 50000:
            err('full 50k ImageNet validation required')
        if not isinstance(run.get('accuracy_top5'),(float,int)): err('ImageNet top-5 missing')
        if run.get('batch_size') != 256: err('ImageNet paper global batch 256 required')
        if run.get('kind')=='resnet' and run.get('depth') not in (18,34,50,101,152):
            err('unsupported ImageNet paper depth')
        if run.get('paper_deviations'): err('declared ImageNet deviations prohibit strict reproduction')
    elif family == 'detection':
        if data.get('name') not in ('COCO2014','PASCAL_VOC_2007_2012'): err('wrong detection dataset')
        if not isinstance(run.get('map'), (int,float)): err('detection mAP missing')
        if run.get('paper_deviations'): err('declared detection deviations prohibit strict reproduction')
    else: err('unrecognized experiment family')
    return problems


def inspect(manifest: str | Path, scope: str = 'cifar10_paper',
            checkpoint_root: str | Path | None = None) -> dict:
    if scope not in SCOPES and scope != 'full_paper':
        raise ValueError('scope must be ' + ', '.join([*SCOPES,'full_paper']))
    p = Path(manifest)
    obj = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(obj, dict): raise ValueError('Evidence manifest must be an object')
    runs = obj.get('runs', [])
    if not isinstance(runs, list): raise ValueError('runs must be a list')
    checked = []
    for i,r in enumerate(runs):
        errors = _valid_run(r,scope)
        if checkpoint_root is not None and isinstance(r,dict):
            ident=r.get('run_id','')
            if not isinstance(ident,str) or not ident or any(z in ident for z in ('/','\\','..')):
                errors.append('unsafe checkpoint identifier')
            else:
                candidate=Path(checkpoint_root)/(ident+'.pt')
                if not candidate.is_file():
                    errors.append('checkpoint file missing for independent SHA256 comparison')
                else:
                    digest=hashlib.sha256()
                    with candidate.open('rb') as f:
                        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
                    if digest.hexdigest()!=r.get('checkpoint_sha256'):
                        errors.append('checkpoint file SHA256 mismatch')
        checked.append({'index':i,'family':r.get('family') if isinstance(r,dict) else None,
                        'kind':r.get('kind') if isinstance(r,dict) else None,
                        'depth':r.get('depth') if isinstance(r,dict) else None,
                        'passed':not errors,'violations':errors})
    good = [r for r,row in zip(runs,checked) if row['passed']]
    covered = {'cifar10':set(), 'imagenet':set(), 'detection':set()}
    for r in good:
        fam = _get_family(r)
        if fam == 'detection':
            covered[fam].add((str(r.get('kind')), 'coco' if r['dataset']['name']=='COCO2014' else 'voc'))
        elif fam in ('cifar10','imagenet'):
            covered[fam].add((str(r.get('kind')),r.get('depth')))
    required = {'cifar10_paper':['cifar10'],'imagenet_paper':['imagenet'],
                'detection_paper':['detection'],'full_paper':['cifar10','imagenet','detection']}.get(scope,[])
    incomplete=[]
    for fam in required:
        miss=SCOPES[f'{fam}_paper'] - covered[fam]
        for pair in sorted(miss,key=str): incomplete.append({'family':fam,'missing':list(pair)})
    single = scope=='cifar10_single' and bool(covered['cifar10'])
    passed=(single or (bool(required) and not incomplete))
    if scope in ('cifar10_paper','full_paper'):
        resnet110_seeds={r.get('seed') for r in good if _get_family(r)=='cifar10' and r.get('kind')=='resnet' and r.get('depth')==110}
        if len(resnet110_seeds)<5:
            incomplete.append({'family':'cifar10','missing':['resnet',110,'5_independent_seeds_required']})
            passed=False
    if scope=='cifar10_single': stage='single_real_experiment_complete' if passed else 'not_complete'
    elif passed: stage='scope_protocol_complete_not_independently_reproduced'
    elif any(covered.values()): stage='partial_real_dataset_evidence'
    else: stage='implementation_or_unverified'
    if scope=='full_paper':
        passed=False
        stage='external_independent_review_required_for_full_paper'
        incomplete.append({'family':'full_paper','missing':['ILSVRC_competition_detection_localization',
                    'COCO_detection_improvements_and_ensemble','external_scientific_review']})
    return {'scope':scope,'passed':passed,'stage':stage,'runs_total':len(runs),
            'runs_valid':len(good),'coverage':{k:sorted([list(z) for z in v],key=str) for k,v in covered.items()},
            'missing':incomplete,'run_checks':checked,
            'checkpoint_integrity': 'locally_verified' if checkpoint_root is not None else 'self_reported_only',
            'important_limit':'Evidence is self-reported by the training pipeline; this does not independently establish scientific replication, metric agreement, or absence of dataset leakage.'}
