from dataclasses import dataclass, field
import pandas as pd
import numpy_financial as npf
import numpy as np

@dataclass
class Loan:
    wac: float  # Weighted Average Coupon (annual interest rate)
    wam: int    # Weighted Average Maturity (in months)
    pv: float   # Present Value (loan amount)
    #pv is B0


@dataclass
class Scenario:
    # loan: Loan
    smmV: np.ndarray # Single Monthly Mortality (prepayment vector)
    dqV: np.ndarray # Delinquency rate
    mdrV: np.ndarray # Monthly Default Rate
    sevV: np.ndarray # Severity
    recovery_lag: int # recovery_lag
    refund_smm: np.ndarray # refund_smm, treat as prepay
    premium_discount: float   # premium of discount

    
    
@dataclass
class Yield:
    # scenario: Scenario
    yieldValue: float
    fullpx: float = 0


def safedivide(a, b):
    if np.isclose(b, 0, rtol=0, atol=3e-11):
        return 0
    else:
        return a / b
def calc(v):
    numerator = np.sum(np.maximum(0.0, v) * (np.arange(1, len(v) + 1) / 12.))
    denominator = np.sum(np.maximum(0.0, v))
    return safedivide(numerator, denominator) 
def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result       


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
        recovery_lag = self.scenario.recovery_lag
        refund_smm = self.scenario.refund_smm
        premium_discount = self.scenario.refund_smm
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

        # adding premium_discount
        refund_smm = refund_smm * premium_discount

        # Initialize survivorship
        survivorshipV = np.insert(np.cumprod(np.ones(wam) - smmV - refund_smm - mdrV), 0, 1) # len: wam+1

        # SMM actual balance
        actualBalanceV = survivorshipV * balancesV # len: wam+1, ending interest bearing balance

        # Shifted actual balance (starts at pv)
        b_balanceV= actualBalanceV[:-1] # len: wam

        actualBalanceV = actualBalanceV[1:] # len: wam, ending interest bearing balance
        #balance difference, balance minus previous month balance, the balance remove last month, previous month balance remove 1st month, then shift to left, 
        BalanceDiffV = b_balanceV - actualBalanceV

        # Actual interest
        actInterestV = b_balanceV * rate * (1-dqV)

        # Scheduled, prepayment, default, and total principals
        schedPrinV = survivorshipV[:-1] * principalsV[:-1] * (1-dqMdrV) # maximum between dq/mdr
        dqPrinV = survivorshipV[:-1] * principalsV[:-1] * np.maximum(0, dqV - mdrV) # deadbeat balance
        prepayPrinV = survivorshipV[:-1] * balancesV[1:] * smmV
        defaultPrinV = b_balanceV * mdrV
        # new defaultPrinV = total default times default_timingV
        #Total_default_p * pv (B0), Total_default= otal_default_p * pv (B0),
        # defaultPrinV need shift recovery_lag, defaultPrinV=defaultPrinV[:reco]
        totalEndingBalV = actualBalanceV + dqPrinV 
        totalBeginningBalV = b_balanceV.copy()
        totalBeginningBalV[1:] = dqPrinV[:-1]

        # Losses, recoveries, and writedowns
        lossV = defaultPrinV * sevV
        recoveryV = defaultPrinV - lossV
        #shift recovery_lag
        recoveryV = shift_elements(recoveryV, recovery_lag, 0)
       # Test shift_elements(arr, 2, 0)
       #arr = np.array([1, 2, 3, 4, 5, 6])
       # shifted_arr = shift_elements(arr, 2, 0)
       #print(shifted_arr)     
        
        writedownV = b_balanceV * sevV * mdrV

        # Total principal and cash flow
        totalPrinV = schedPrinV + prepayPrinV + recoveryV
        # Recovery Lag section
        #  totalPrinV = schedPrinV + prepayPrinV 
        # length totalPrinV by recovery_lag months
        #  totalPrinV[recovery_lag:] += recoveryV

        # refund section
        # refundPrinV = survivorshipV[:-1] * balancesV[1:] * refund_smmV
        # totalPrinV = schedPrinV + prepayPrinV  + refundPrinV * refund_premium
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

        wal_cfl=calc(cflV) 
        wal_PrinV=calc(totalPrinV)    
        wal_InterestV=calc(actInterestV) 
        wal_BalanceDiffV=calc(BalanceDiffV)     

        return self.resultDF
    
    def getPX(self):
        return self.resultPX
    

if __name__ == '__main__':
    loan = Loan(wac=0.0632, wam=357, pv=100000000)
  
    # Define vectors for the scenario
    smmVec = np.ones(loan.wam) * 0.01
    dqVec = np.ones(loan.wam) * 0.1
    mdrVec = np.ones(loan.wam) * 0.1
    sevVec = np.ones(loan.wam) * 0.2
    recovery_lagValue = 2
    refund_smmVec = np.ones(loan.wam) * 00.5
    premium_discountValue = 0.1
   
    scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec, 
                        recovery_lag=recovery_lagValue, refund_smm = refund_smmVec, premium_discount = premium_discountValue)
    y = Yield(yieldValue=0.055)
    x = Yield(yieldValue=0.0632)

    output = Output(loan=loan, scenario=scenario, px=y)
    output1 = Output(loan=loan, scenario=scenario, px=x)

    df = output.getCashflow()
    px = output.getPX()

    print(df)
    print(px)