import json, glob, os
rows=[]
for p in sorted(glob.glob('results/*/result.json')):
    d=json.load(open(p)); a=d.get('args',{})
    rows.append([os.path.basename(os.path.dirname(p)), a.get('head'), a.get('ema_mode'),
                 a.get('seed'), a.get('snap_every'), a.get('epochs'),
                 round(d.get('val_mae_live',-1),4), round(d.get('val_mae_ema',-1),4)])
hdr=['run','head','mode','seed','snap','ep','live','ema']
print(' | '.join(f'{h}' for h in hdr))
for r in rows:
    print(' | '.join(str(x) for x in r))
