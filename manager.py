import sys
import subprocess

loss = 'zhen'
var = 0.04
d_arr = [5., 10., 15., 20., 25., 30.]

procs = []
for d in d_arr:
    proc = subprocess.Popen([sys.executable, 'experiments.py', '{}'.format(loss), '{}'.format(d), '{}'.format(var), 'train', '3'])
    procs.append(proc)

for proc in procs:
    proc.wait()