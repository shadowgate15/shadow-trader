"""Build loop_cy.pyx in a temp dir and time it against the same data as bench_loop.py."""
import sys, time, tempfile, shutil, subprocess, os, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(here, "loop_cy.pyx"), tmp)
t = time.perf_counter()
subprocess.run([sys.executable, "-m", "cython", "-3", "loop_cy.pyx"], cwd=tmp, check=True)
import sysconfig
inc = sysconfig.get_paths()["include"]
ext = sysconfig.get_config_var("EXT_SUFFIX")
subprocess.run(["cc", "-O2", "-shared", "-undefined", "dynamic_lookup", "-I", inc, "loop_cy.c", "-o", "loop_cy" + ext], cwd=tmp, check=True)
print(f"cython+cc build: {time.perf_counter()-t:.2f} s")
sys.path.insert(0, tmp)
sys.argv = ["x", "5000000"]
sys.path.insert(1, here)
import bench_loop as b
import loop_cy
secs, out = b.timeit(loop_cy.loop_cy, b.open_, b.high, b.low, b.close, b.sig, repeat=5)
b.report("cython (cc -O2) typed memoryviews", b.N, secs, out)
print("bitwise equals numba/py:", out == b.loop_nb(b.open_, b.high, b.low, b.close, b.sig))
