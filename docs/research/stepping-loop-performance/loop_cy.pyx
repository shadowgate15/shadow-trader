# cython: boundscheck=False, wraparound=False, cdivision=True
def loop_cy(double[::1] o, double[::1] h, double[::1] l, double[::1] c, signed char[::1] s, double STOP=0.5, double TGT=0.8):
    cdef Py_ssize_t i, n = o.shape[0]
    cdef int pos = 0, pending = 0
    cdef long trades = 0
    cdef double entry = 0.0, cash = 10000.0, peak = 10000.0, maxdd = 0.0, eq, dd
    for i in range(n):
        if pending != 0:
            pos = pending; entry = o[i]; pending = 0; trades += 1
        if pos != 0:
            if pos > 0:
                if l[i] <= entry - STOP:
                    cash += (-STOP) * pos; pos = 0
                elif h[i] >= entry + TGT:
                    cash += TGT * pos; pos = 0
            else:
                if h[i] >= entry + STOP:
                    cash += (-STOP) * (-pos); pos = 0
                elif l[i] <= entry - TGT:
                    cash += TGT * (-pos); pos = 0
        if pos == 0 and s[i] != 0:
            pending = s[i]
        eq = cash + (c[i] - entry) * pos if pos != 0 else cash
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > maxdd: maxdd = dd
    return cash, maxdd, trades
