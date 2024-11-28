from dataclasses import dataclass, field
import pandas as pd
import numpy_financial as npf
import numpy as np

@dataclass
class Loan:
    wac: float  # Weighted Average Coupon (annual interest rate)
    wam: int    # Weighted Average Maturity (in months)
    pv: float   # Present Value (loan amount)


@dataclass
class Scenario:
    # loan: Loan
    smmV: np.ndarray # Single Monthly Mortality (prepayment vector)
    dqV: np.ndarray # Delinquency rate
    mdrV: np.ndarray # Monthly Default Rate
    sevV: np.ndarray # Severity
    
    
@dataclass
class Yield:
    # scenario: Scenario
    yieldValue: float
    fullpx: float = 0

@dataclass
class Output:
    loan: Loan
    scenario: Scenario
    px: Yield
    resultDF: pd.DataFrame = field(default_factory=pd.DataFrame) # empty dataframe
    resultPX: float = 0

    def getCashflow(self):
        wac = self.loan.wac
        wam = self.loan.wam
        pv = self.loan.pv

        smmV = self.scenario.smmV
        dqV = self.scenario.dqV
        mdrV = self.scenario.mdrV
        sevV = self.scenario.sevV

        dqMdrV = np.maximum(dqV, mdrV)

        yieldValue = self.px.yieldValue

        # Amortization

        rate = wac / 12
        X = npf.pmt(rate, wam, pv) # Fixed monthly payment
        
        # All vectors are wam length long, except survivorship, balances, and specifically noted balance (len: wam + 1)
        monthsV = np.arange(1, wam + 1)
        balancesV = pv * (1 - (1 + rate) ** -(wam - np.arange(wam + 1))) / (1 - (1 + rate)**-wam) # len: wam+1
        paymentsV = np.ones(wam) * abs(X)
        interestsV = balancesV * rate  # len: wam+1
        principalsV = abs(X) - interestsV  # len: wam+1

        #Scenario

        # Initialize survivorship
        survivorshipV = np.insert(np.cumprod(np.ones(wam) - smmV - mdrV), 0, 1) # len: wam+1

        # SMM actual balance
        actualBalanceV = survivorshipV * balancesV # len: wam+1, ending interest bearing balance

        # Shifted actual balance (starts at pv)
        b_balanceV= actualBalanceV[:-1] # len: wam

        actualBalanceV = actualBalanceV[1:] # len: wam, ending interest bearing balance

        # Actual interest
        actInterestV = b_balanceV * rate * (1-dqV)

        # Scheduled, prepayment, default, and total principals
        schedPrinV = survivorshipV[:-1] * principalsV[:-1] * (1-dqMdrV) # maximum between dq/mdr
        dqPrinV = survivorshipV[:-1] * principalsV[:-1] * np.maximum(0, dqV - mdrV) # deadbeat balance
        prepayPrinV = survivorshipV[:-1] * balancesV[1:] * smmV
        defaultPrinV = b_balanceV * mdrV

        totalEndingBalV = actualBalanceV + dqPrinV 
        totalBeginningBalV = b_balanceV.copy()
        totalBeginningBalV[1:] = dqPrinV[:-1]

        # Losses, recoveries, and writedowns
        lossV = defaultPrinV * sevV
        recoveryV = defaultPrinV - lossV
        writedownV = b_balanceV * sevV * mdrV

        # Total principal and cash flow
        totalPrinV = schedPrinV + prepayPrinV + recoveryV
        cflV = totalPrinV + actInterestV

        # Create scenario DataFrame
        self.resultDF = pd.DataFrame({
            "Months": monthsV,
            "Prin": totalPrinV,
            "SchedPrin": schedPrinV,
            "Prepay Prin": prepayPrinV,
            "Default": defaultPrinV,
            "Writedown": writedownV,
            "Recovery": recoveryV,
            "Interest": actInterestV,
            "Beginning Balance": b_balanceV,
            "Balance": actualBalanceV,
            "CFL": cflV,
        })

        # Yield/PX

        #### LIFT INTO FUNCTION AND PASS IN CASHFLOW #####
        yV = (1 + yieldValue/12)**monthsV
        self.resultPX = np.sum(cflV/yV) / b_balanceV[0]

        return self.resultDF
    
    def getPX(self):
        return self.resultPX
    

loan = Loan(wac=0.0632, wam=357, pv=100000000)
# Define vectors for the scenario
smmVec = np.ones(loan.wam) * 0.01
dqVec = np.ones(loan.wam) * 0.1
mdrVec = np.ones(loan.wam) * 0.1
sevVec = np.ones(loan.wam) * 0.2


scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec)
y = Yield(yieldValue=0.055)
x = Yield(yieldValue=0.0632)

output = Output(loan=loan, scenario=scenario, px=y)
output1 = Output(loan=loan, scenario=scenario, px=x)

df = output.getCashflow()
px = output.getPX()

print(df)
print(px)