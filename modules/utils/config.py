
from multiprocessing import cpu_count
N_JOBS = (cpu_count()) if ((cpu_count()) > 1) else 1

SEED = 7
