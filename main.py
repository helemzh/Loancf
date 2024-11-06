import numpy_financial as npf
import pandas as pd

rate = 0.06 / 12
nper = 360
pv = 100000

# Fixed monthly payment
X = npf.pmt(rate, nper, pv)

# Initialize lists for each row of data
months = list(range(nper + 1))
balances = [pv]
rates = [.06] * (nper + 1)
payments = [abs(X)] * (nper + 1)
interests = []
principals = []

# Generate the payment schedule
balance = pv
for month in range(1, nper + 1):
    interest = balance * rate
    principal = abs(X) - interest
    balance -= principal

    # Append values to lists
    if balance > 0:
        balances.append(balance)
    else:
        balances.append(0)
    interests.append(interest)
    principals.append(principal)
    
# Insert a zero interest and principal for the initial month
interests.insert(0, 0)
principals.insert(0, 0)
payments[0] = 0

# Create a DataFrame
df = pd.DataFrame({
    "Month": months,
    "Balance": balances,
    "Mortgage Rate (WAC)": rates,
    "Total Payment": payments,
    "Interest": interests,
    "Principal": principals
})

print(df)