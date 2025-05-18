from dataclasses import dataclass, field
import pandas as pd
import numpy_financial as npf
import numpy as np

cpr2smm = lambda cpr: 1-(1-cpr)**(1/12)
                       
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
    recovery_lag: int=0 # recovery_lag
    refund_smm: np.ndarray =0 # refund_smm, treat as prepay
    refund_premium: float  =1.0 # premium of discount
    aggMDR: float  =0.0 # mdr value of percentage of B0
    aggMDR_timingV: np.ndarray = 0 #mdr percentage monthly vector
    compIntHC: float = 0
    is_advance: bool = False  #todo: adv | dq+default | dq=default at mon 1to4

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
    wal_PrinV: float = 0   
    wal_BalanceDiffV: float = 0
    wal_InterestV: float = 0
    wal_cfl: float = 0

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
        refund_premium = self.scenario.refund_premium
        dqMdrV = dqV + mdrV  # dqV is additional

        yieldValue = self.px.yieldValue

        # Amortization

        rate = wac / 12
        X = -npf.pmt(rate, wam, pv) # Fixed monthly payment
        
        # All vectors are wam length long, except survivorship, balances, and specifically noted balance (len: wam + 1)
        monthsV = np.arange(1, wam + 1)
        balancesV = pv * (1 - (1 + rate) ** -(wam - np.arange(wam + 1))) / (1 - (1 + rate)**-wam) # len: wam+1
        paymentsV = np.ones(wam) * X
        interestsV = balancesV[:-1] * rate  # len: wam
        principalsV = X - interestsV  # len: wam
        paydownV = principalsV / balancesV[:-1]
        

        p_survV = np.cumprod(np.ones(wam) - smmV - refund_smm - mdrV)
        default_aggMDRV = pv*self.scenario.aggMDR * self.scenario.aggMDR_timingV
        dqPrin_aggMDRV = paydownV * default_aggMDRV
        scaled_default_aggMDRV = default_aggMDRV / ( balancesV[:-1] * p_survV )
        cum_scaled_default_aggMDRV = np.cumsum(scaled_default_aggMDRV)
        survivorshipV=np.insert(p_survV*(1-cum_scaled_default_aggMDRV),0,1)#N+1

        actualBalanceV = survivorshipV * balancesV # len: wam+1
        b_balanceV= actualBalanceV[:-1] # len: wam, beginning int bearing bal
        actualBalanceV = actualBalanceV[1:] # len: wam, ending interest bearing balance
        balanceDiffV = b_balanceV - actualBalanceV       

        # Scheduled, prepayment, default, and total principals
        # deadbeat balance
        dqPrinV = np.zeros(wam) if self.scenario.is_advance else (
            survivorshipV[:-1] * principalsV * dqMdrV + dqPrin_aggMDRV )
        schedPrinV = survivorshipV[:-1] * principalsV - dqPrinV
        prepayPrinV = survivorshipV[:-1] * balancesV[1:] * smmV
        defaultV = b_balanceV * mdrV + default_aggMDRV
        totalEndingBalV = actualBalanceV + dqPrinV 
        totalBeginningBalV = b_balanceV.copy()
        totalBeginningBalV[1:] = dqPrinV[:-1]

        # Losses, recoveries, and writedowns
        writedownV = defaultV * sevV
        recoveryV = defaultV - writedownV
        recoveryV = shift_elements(recoveryV, recovery_lag, 0)
        
        # Total principal and cash flow
        refundPrinV = survivorshipV[:-1] * balancesV[1:] * refund_smm
        totalPrinV = schedPrinV + prepayPrinV + recoveryV
        compIntV = prepayPrinV * rate * self.scenario.compIntHC
        refundIntV = refundPrinV * rate
        
            
        actInterestV = rate*b_balanceV if self.scenario.is_advance else (
            rate*(b_balanceV*(1-dqMdrV) - default_aggMDRV) - compIntV)
        actInterestV -= refundIntV
        cflV = totalPrinV + actInterestV

        # Create scenario DataFrame
        self.resultDF = pd.DataFrame({
            "Months": monthsV,
            "Prin": totalPrinV,
            "SchedPrin": schedPrinV,
            "Prepay Prin": prepayPrinV,
            "Default": defaultV,
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
        refundCflV = refundPrinV * refund_premium

        self.resultPX = np.sum(cflV/yV) / (b_balanceV[0]-np.sum(refundPrinV/yV))

 
        wal_PrinV=calc(totalPrinV)    
        wal_BalanceDiffV=calc(balanceDiffV)
        wal_InterestV=calc(actInterestV)   
        wal_cfl=calc(cflV)   
        
        self.wal_PrinV=wal_PrinV   
        self.wal_BalanceDiffV=wal_BalanceDiffV
        self.wal_InterestV=wal_InterestV
        self.wal_cfl=wal_cfl   
        
        return self.resultDF
    
    def getPX(self):
        return self.resultPX
    def get_wal_PrinV(self):
        return self.wal_PrinV
    
    def get_wal_BalanceDiffV(self):
        return self.wal_BalanceDiffV 
    def get_wal_cfl(self):
        return self.wal_cfl
    
    def get_wal_InterestV(self):
        return self.wal_InterestV

if __name__ == '__main__':
   # loan = Loan(wac=0.0632, wam=357, pv=100000000)
    loan = Loan(wac=0.06, wam=24, pv=100000)
  
    # Define vectors for the scenario
    smmVec = np.ones(loan.wam) * 0.01
    dqVec = np.ones(loan.wam) * 0.1
    #mdrVec = np.ones(loan.wam) * 0.1
    mdrVec = np.ones(loan.wam) * 0.0
    sevVec = np.ones(loan.wam) * 0.2
  #  aggMDR_timing_Vec = np.ones(loan.wam) * 0
  #  aggMDR_timing_Vec=np.insert(aggMDR_timing_Vec,0,0.1)
    aggMDR_Value = 0.1
    #x = np.arange(loan.wam, dtype=float)
    x = np.zeros(loan.wam, dtype=float)
    np.full_like(x, 0.01)
    aggMDR_timing_Vec = np.full_like(x, 0.1)
 
    recovery_lagValue = 0
   
    #scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec, 
    #                    recovery_lag=recovery_lagValue, refund_smm = refund_smmVec, premium_discount = premium_discountValue)
    scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)
    #scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec, recovery_lag=recovery_lagValue)
    y = Yield(yieldValue=0.055)
    x = Yield(yieldValue=0.0632)

    output = Output(loan=loan, scenario=scenario, px=y)
    output1 = Output(loan=loan, scenario=scenario, px=x)

    df = output.getCashflow()
    px = output.getPX()

    print(df)
    print(px)
