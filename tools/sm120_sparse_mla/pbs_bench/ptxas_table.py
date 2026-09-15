#!/usr/bin/env python3
import json, collections, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(len(rows), "MG kernel instantiations")
by=collections.defaultdict(dict)
for r in rows: by[r["kernel"]][r["variant"]]=r
variants=sorted({r["variant"] for r in rows}, key=lambda v:(v!="pristine",v))
print("%-72s" % "kernel", " | ".join("%16s" % v for v in variants))
for k in sorted(by):
    cells=[]
    for v in variants:
        r=by[k].get(v)
        cells.append("%16s" % ("-" if r is None else "r%d s%d/%d k%d" % (r["regs"], r["spill_st"], r["spill_ld"], r["stack"])))
    print("%-72s" % k[:72], " | ".join(cells))
