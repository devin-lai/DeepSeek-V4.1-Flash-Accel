#!/usr/bin/env python3
"""Run every built variant over the shape x page-size grid; append JSON lines."""
import json, subprocess, sys, pathlib, itertools, os
out=pathlib.Path(sys.argv[1]); variants=sys.argv[2:]
res=out/'results.jsonl'
SINGLE=[('bf16',8,128,4096,'window'),('bf16',16,128,4096,'window'),('fp8',32,512,4096,'random'),('fp8',64,2048,2048,'random'),('fp8',128,2048,2048,'random')]
DUAL=[('bf16',8,128,4096,512),('bf16',32,128,4096,512)]
def pbs_list(v): return [32,64] if v in ('pristine','mode0') or v.startswith('mode0') else [16,32,64,128]
with res.open('a') as f:
    for v in variants:
        b=out/'bin'/f'bench_mg_{v}'
        if not b.exists(): print('missing',b); continue
        for (cm,nh,topk,tokens,idx) in SINGLE:
            for pbs in pbs_list(v):
                cmd=[str(b),'--cm',cm,'--nh',str(nh),'--topk',str(topk),'--pbs',str(pbs),'--tokens',str(tokens),'--idx',idx,'--iters','20']
                r=subprocess.run(cmd,capture_output=True,text=True)
                line=r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({'error':r.stderr[-300:]})
                d=json.loads(line); d['variant']=v; f.write(json.dumps(d)+'\n'); f.flush()
                print(v,cm,nh,topk,pbs,d.get('us_median'),d.get('hash_out','')[:8],d.get('error',''))
        for (cm,nh,topk,tokens,xk) in DUAL:
            for pbs in pbs_list(v):
                cmd=[str(b),'--dual','1','--cm',cm,'--nh',str(nh),'--topk',str(topk),'--pbs',str(pbs),'--tokens',str(tokens),'--idx','window','--extra_topk',str(xk),'--extra_pbs','64','--iters','20']
                r=subprocess.run(cmd,capture_output=True,text=True)
                line=r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({'error':r.stderr[-300:]})
                d=json.loads(line); d['variant']=v; f.write(json.dumps(d)+'\n'); f.flush()
                print(v,'dual',cm,nh,topk,pbs,d.get('us_median'),d.get('hash_out','')[:8],d.get('error',''))
