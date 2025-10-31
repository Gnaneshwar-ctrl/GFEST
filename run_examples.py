
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[0]
ex = root/'examples'
res = root/'results'
res.mkdir(exist_ok=True, parents=True)

subprocess.check_call([sys.executable, str(ex/'generate_synthetic.py')])

subprocess.check_call([sys.executable, str(root/'gfest'/'run_gfest.py'),
    '--csv', str(ex/'michaelis_menten.csv'),
    '--y', 'y', '--x', 'x',
    '--reg', 'lasso', '--pop', '30', '--gens', '15', '--terms', '12', '--seed', '1',
    '--outdir', str(res/'mm_lasso'),
    '--sg_window', '11', '--sg_poly', '3'
])

subprocess.check_call([sys.executable, str(root/'gfest'/'run_gfest.py'),
    '--csv', str(ex/'second_order.csv'),
    '--y', 'y', '--x', 'A', 'B',
    '--reg', 'ols', '--pop', '30', '--gens', '15', '--terms', '12', '--seed', '2',
    '--outdir', str(res/'second_order_ols')
])

print("Done. Results in", res)
