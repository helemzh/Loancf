import numpy_financial as npf

rate = 0.06 / 12
nper = 360
pv = 100000

X = npf.pmt(rate, nper, pv)
print(X)