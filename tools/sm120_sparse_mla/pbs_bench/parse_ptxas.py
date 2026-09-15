#!/usr/bin/env python3
"""Summarise `ptxas -v` logs: registers / spills per MG prefill kernel instantiation."""
import re, subprocess, sys, json, pathlib
def demangle(n):
    try: return subprocess.run(['/usr/local/cuda/bin/cu++filt', n], capture_output=True, text=True).stdout.strip() or n
    except Exception: return n
rows=[]
for log in sys.argv[1:]:
    txt=pathlib.Path(log).read_text()
    for m in re.finditer(r"Compiling entry function '([^']+)' for 'sm_\w+'\s*\nptxas info\s*: Function properties for \1\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s*\nptxas info\s*: Used (\d+) registers", txt):
        name=demangle(m.group(1))
        if 'prefill_mg' not in name: continue
        short=re.sub(r'flashinfer::sparse_mla_sm120::','',name)
        short=re.sub(r'\(ModelType\)1','DSV4',short); short=re.sub(r'\(ComputeMode\)0','FP8',short); short=re.sub(r'\(ComputeMode\)1','BF16',short)
        short=re.sub(r'\(int\)','',short); short=re.sub(r'\(bool\)','',short); short=short.split('(')[0]
        rows.append(dict(variant=pathlib.Path(log).stem, kernel=short, stack=int(m.group(2)), spill_st=int(m.group(3)), spill_ld=int(m.group(4)), regs=int(m.group(5))))
for r in rows: print(json.dumps(r))
