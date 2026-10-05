"""Real BDDX CPU integration: LK is the only changed batch tensor."""
import json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict

project=Path('/mnt/pool/syf/asuad_lk_20260920')
os.chdir(str(project));sys.path.insert(0,str(project))
out=project/'output/BDDX_eden2_lk_fresh_testing_20260920'
old=Path('/mnt/pool/syf/asuad_repaired_20260918/output/BDDX_eden2_fresh_testing_20260919')
from src.layers.bert import BertTokenizer
from src.datasets.vl_dataloader import build_dataset
from src.modeling.visual_references import VisualReferenceExtractor
args=EasyDict(json.loads((old/'log/args.json').read_text()))
args.update(json.loads((out/'config.json').read_text()))
tokenizer=BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
records=[]
for split,train,expected in [('training',True,21143),('testing',False,2859)]:
    lk=build_dataset(args,'BDDX/'+split+'_32frames.yaml',tokenizer,is_train=train)
    dis_args=EasyDict(dict(args));dis_args.vision_flow_backend='dis'
    dis_args.flow_cache_dir='datasets/BDDX/flow_cache_dis112_v1'
    dis=build_dataset(dis_args,'BDDX/'+split+'_32frames.yaml',tokenizer,is_train=train)
    assert len(lk)==len(dis)==expected
    for i in [0,1,2,3,17,63]:
        for module in (random,np.random,torch):
            if module is random: module.seed(88+i)
            elif module is np.random: module.seed(88+i)
            else: module.manual_seed(88+i)
        a=lk[i]
        rng_a=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        random.seed(88+i);np.random.seed(88+i);torch.manual_seed(88+i)
        b=dis[i]
        assert a[0]==b[0]
        assert len(a[1])==len(b[1])
        assert all(torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
                   for x,y in zip(a[1][:-1],b[1][:-1])), 'Non-flow batch tensor changed'
        assert a[2]['augmentation_geometry']==b[2]['augmentation_geometry']
        assert random.getstate()==rng_a[0]
        assert np.array_equal(np.random.get_state()[1],rng_a[1][1])
        assert torch.equal(torch.get_rng_state(),rng_a[2])
        motion=a[1][-1]
        assert tuple(motion.shape)==(32,4) and torch.isfinite(motion).all()
        assert (motion[0]==0).all() and (motion[:,3].abs()<=1.000001).all()
        records.append(dict(split=split,index=i,key=a[0],shape=list(motion.shape),
            mean_abs=motion.abs().mean(0).tolist(),old_dis_mean_abs=b[1][-1].abs().mean(0).tolist(),
            geometry=a[2]['augmentation_geometry'],nonzero_frames=int((motion.abs().sum(1)>0).sum())))
try:
    from types import SimpleNamespace
    VisualReferenceExtractor._motion(SimpleNamespace(flow_backend='lk'),None)
except RuntimeError as error:
    assert 'Offline LK features are mandatory' in str(error)
else:
    raise AssertionError('LK path silently fell back to an online estimator')
result=dict(status='passed',dataset='BDDX',training=21143,testing=2859,samples=records,
    unchanged='RGB crop/tensor, caption masking/labels, attention input mask, metadata and RNG state all identical to DIS for 12 paired real examples',
    gpu_used=False,lk_fallback_rejected=True)
(out/'lk_dataset_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
