import numpy_financial as npf
import pandas as pd
import numpy as np

wam = 24 # months
wala = wam + 1
wac = 0.06
rate = wac / 12
pv = 100000

# Fixed monthly payment
X = npf.pmt(rate, wam, pv)

# Initialize lists for each row of data
months = np.arange(wala)
balances =  pv * (1 - (1 + wac/12.) ** -(wam - np.arange(wala))) / (1 - (1+wac/12.)**-wam)
rates = np.ones(wala) * wac
payments = np.ones(wala) * abs(X)

interests = np.zeros_like(balances)
interests[1:] = balances[:-1] * rate

principals = abs(X) - interests

interests[0] = 0
principals[0] = 0
payments[0] = 0

# Mortgage Amortization
amortization = pd.DataFrame({
    "Month": months,
    "Balance": balances,
    "Mortgage Rate (WAC)": rates,
    "Total Payment": payments,
    "Interest": interests,
    "Principal": principals
})

#smm/survivorship
smmVector = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
survivorship = np.ones(wala)
survivorship = np.cumprod(survivorship * (np.ones(wala) - smmVector))

balancesTemp = np.zeros(wala)
balancesTemp[:-1] = balances[1:]
actualBalance = survivorship * balancesTemp

survivorshipTemp = np.ones(wala)
survivorshipTemp[1:] = survivorship[:-1]

beginningBalance = np.ones(wala) * 100000
beginningBalance[1:] = actualBalance[:-1]

prepayPrin = survivorshipTemp * balancesTemp * smmVector

amortPrinTemp = np.ones(wala)
amortPrinTemp[:-1] = principals[1:]
schedPrin = survivorshipTemp * amortPrinTemp

balanceCheck = beginningBalance - actualBalance

actInterest = beginningBalance * rate

smm = pd.DataFrame({
    "smm vector": smmVector,
    "survivorship": survivorship,
    "b_bal": beginningBalance,
    "actual balance": actualBalance,
    "prepay prin": prepayPrin,
    "sched prin": schedPrin,
    "balance chk": balanceCheck,
    "actual interest": actInterest
})

# print(amortization)
print(smm)