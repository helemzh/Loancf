# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

from dataclasses import dataclass, field
import pandas as pd
import numpy_financial as npf
import numpy as np
from scipy.optimize import brentq
from typing import Optional

# --- Utility Functions ---

cpr2smm = lambda cpr: 1-(1-cpr)**(1/12)

bey2y = lambda y: 12 * ((1 + y / 2) ** (1 / 6) - 1)
y2bey = lambda y: 2 *((1 + y/12) ** 6 - 1)

def safedivide(a, b):
    if np.isclose(b, 0, rtol=0, atol=3e-11):
        return 0
    else:
        return a / b

def calc(v): # for WAL calculations
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

def pad_zeros(vec, n, pad_value=0): # for lag
    if len(vec) < n:
        if pad_value == 'last':
            pad_val = vec[-1]
        else:
            pad_val = pad_value
        return np.concatenate([vec, np.full(n - len(vec), pad_val)])
    return vec


# --- Data Classes ---

@dataclass
class Config:
    servicing_fee_method: str="avg"  # or "beg", toggle between avg and beginning bal servicing fee calculation
    rate_red_method: bool=False # default unfixed rate
    is_advance: bool=False  #todo: adv | dq+default | dq=default at mon 1to4

@dataclass
class Scenario:
    smmV: np.ndarray # Single Monthly Mortality (prepayment vector)
    dqV: np.ndarray # Delinquency rate
    mdrV: np.ndarray # Monthly Default Rate
    sevV: np.ndarray # Severity
    dq_adv_prin: Optional[np.ndarray] = field(default=None) # Fraction of dq principal recovered immediately
    dq_adv_int: Optional[np.ndarray] = field(default=None) # Fraction of dq interest recovered immediately
    rate_redV: Optional[np.ndarray] = field(default=None) # reduces monthly rate
    refund_smm: Optional[np.ndarray] = field(default=None) # treat as prepay
    aggMDR_timingV: Optional[np.ndarray] = field(default=None) # mdr percentage monthly vector
    recovery_lag: int=0
    refund_premium: float=1.0 # premium of discount
    aggMDR: float=0.0 # mdr value of percentage of B0
    compIntHC: float=0.0 # Haircut
    servicing_fee: float=0.0

    def __post_init__(self):
        """Initialize optional arrays to zeros if not provided"""
        if self.refund_smm is None:
            self.refund_smm = np.zeros_like(self.smmV)  # Zero array matching smmV shape
        if self.aggMDR_timingV is None:
            self.aggMDR_timingV = np.zeros_like(self.smmV)
        if self.rate_redV is None:
            self.rate_redV = np.zeros_like(self.smmV)
        if self.dq_adv_prin is None:
            self.dq_adv_prin = np.zeros_like(self.smmV)
        if self.dq_adv_int is None:
            self.dq_adv_int = np.zeros_like(self.smmV)
        

@dataclass
class Input:
    yieldValue: float=0.0
    fullpx: float=0.0


@dataclass
class Loan:
    wac: float  # Weighted Average Coupon (annual interest rate)
    wam: int    # Weighted Average Maturity (in months)
    pv: float   # Present Value (loan amount), pv is B0

    def getCashflow(self, scenario, config):
        wac = self.wac
        wam = self.wam
        pv = self.pv

        rate_redV = scenario.rate_redV
        smmV = scenario.smmV
        dqV = scenario.dqV
        mdrV = scenario.mdrV
        sevV = scenario.sevV
        dq_adv_prin = scenario.dq_adv_prin
        dq_adv_int = scenario.dq_adv_int
        recovery_lag = scenario.recovery_lag
        period_with_lag = wam + recovery_lag # adjust number of rows to add lag
        refund_smm = scenario.refund_smm
        refund_premium = scenario.refund_premium
        #dqMdrV = dqV + mdrV  # dqV is additional, separating the use of dq and mdr

        '''
        Most vectors are wam length.
        survivorship, balances, and specifically noted balance (len: wam + 1)
        relating to servicing fee (len: wam + lag)
        '''
        wacV = wac - rate_redV
        rate = wac/12
        rateV = wacV / 12
        monthsV = np.arange(1, wam + 1 + recovery_lag) # len: wam+lag

        if np.isclose(rate, 0, rtol=0, atol=1e-8):
            # Zero rate case
            balancesV = np.linspace(pv, 0, wam+1) # len: wam+1
            principalsV = np.full(wam, pv / wam)
            interestsV = np.zeros(wam)
            paydownV = principalsV / balancesV[:-1]
        else:
            if config.rate_red_method == False:
                # Fixed rate calculation
                X = -npf.pmt(rate, wam, pv) # Fixed monthly payment
                balancesV = pv * (1 - (1 + rate) ** -(wam - np.arange(wam + 1))) / (1 - (1 + rate)**-wam) # len: wam+1, fixed rate
                interestsV = balancesV[:-1] * rate
                principalsV = X - interestsV
                paydownV = principalsV / balancesV[:-1]
            else:
                # Unfixed rate calculation
                rateV = np.append(rateV, rateV[-1]) #len:wam+1 for balancesV calculation
                denom = (1 + rateV) ** (wam - np.arange(wam + 1)) - 1
                # alphaV = np.where(np.abs(denom) < 1e-12, 0.0, rateV / denom)      
                alphaV = np.divide(rateV, denom, out=np.zeros_like(rateV), where=np.abs(denom) >= 1e-12)
                balancesV = np.concatenate(([pv], pv * np.cumprod(1-alphaV)[:-1]))
                balancesV = np.maximum(balancesV, 0)
                principalsV = balancesV[:-1] - balancesV[1:]
                interestsV = balancesV[:-1] * rateV[:-1]

                paydownV = np.where(balancesV[:-1] != 0, principalsV / balancesV[:-1], 0.0)
                rateV = rateV[:-1]
        
        p_survV = np.cumprod(np.ones(wam) - smmV - refund_smm - mdrV)
        default_aggMDRV = pv*scenario.aggMDR * scenario.aggMDR_timingV
        dqPrin_aggMDRV = paydownV * default_aggMDRV
        scaled_default_aggMDRV = np.where(( balancesV[:-1] * p_survV ) != 0, default_aggMDRV / ( balancesV[:-1] * p_survV ), 0.0)
        cum_scaled_default_aggMDRV = np.cumsum(scaled_default_aggMDRV)
        survivorshipV=np.insert(p_survV*(1-cum_scaled_default_aggMDRV),0,1) # wam+1
        survivorshipV = np.maximum(survivorshipV, 0)

        actualBalanceV = survivorshipV * balancesV # len: wam+1
        b_balanceV= actualBalanceV[:-1] # beginning int bearing bal
        actualBalanceV = actualBalanceV[1:] # ending interest bearing balance

        # Scheduled, prepayment, default, total principals, and deadbeat balance
        schedDQPrinV = survivorshipV[:-1] * principalsV * (1-mdrV) * dqV * (1-dq_adv_prin)
        schedDefaultPrinV = survivorshipV[:-1] * principalsV * mdrV + dqPrin_aggMDRV
        schedPrinV = survivorshipV[:-1] * principalsV - schedDQPrinV - schedDefaultPrinV
        prepayPrinV = survivorshipV[:-1] * balancesV[1:] * smmV
        defaultV = b_balanceV * mdrV + default_aggMDRV
        # totalEndingBalV = actualBalanceV + dqPrinV 
        # totalBeginningBalV = b_balanceV.copy()
        # totalBeginningBalV[1:] = dqPrinV[:-1]

        # Losses, recoveries, and writedowns
        writedownV = defaultV #* sevV
        recoveryV = defaultV - writedownV # old calculation
        recoveryV = shift_elements(recoveryV, recovery_lag, 0) # old calculation
        writedownV = shift_elements(pad_zeros(writedownV, period_with_lag), recovery_lag, 0) # len: wam+lag, added writedownV shift
        recoveryV = writedownV * (1-pad_zeros(sevV, period_with_lag, pad_value='last')) # len: wam+lag, new calculation
        
        refundPrinV = survivorshipV[:-1] * balancesV[1:] * refund_smm
        totalPrinV = pad_zeros(schedPrinV, period_with_lag) + pad_zeros(prepayPrinV, period_with_lag) + recoveryV # len: wam+lag, padded for recovery lag
        compIntV = prepayPrinV * rateV * scenario.compIntHC
        refundIntV = refundPrinV * rateV
        prepayPrinV = survivorshipV[:-1] * balancesV[1:] * smmV + refundPrinV # added refundPrin to prepay calculation

        # Servicing Fee
        defaultBalV = np.maximum(0,np.cumsum(pad_zeros(defaultV, period_with_lag) - writedownV))
        b_totalBalV = pad_zeros(b_balanceV, period_with_lag) + np.insert(defaultBalV, 0, 0)[:-1]
        totalBalV = pad_zeros(actualBalanceV, period_with_lag) + defaultBalV

        servicingFee_rate = scenario.servicing_fee / 12
        servicingFee_begV = b_totalBalV * servicingFee_rate # len: wam+lag, uses beginning balance
        servicingFee_avgV = ((b_totalBalV + totalBalV) / 2) * servicingFee_rate # len: wam+lag
        
        if config.servicing_fee_method == "avg":
            servicingFeeV = servicingFee_avgV
        else:
            servicingFeeV = servicingFee_begV

        # Interest and Cash Flow
        actInterestV = rateV*b_balanceV if config.is_advance else (
            rateV*(b_balanceV * (1-(1-mdrV) *dqV*(1-dq_adv_int) - mdrV) - default_aggMDRV) - compIntV)
        
        actInterestV -= refundIntV
        actInterestV = np.maximum(actInterestV, 0)

        # Padding vectors for result df, len: wam+lag, padded for recovery lag
        cfV = totalPrinV + pad_zeros(actInterestV, period_with_lag)
        totalDefaultV = pad_zeros(schedDQPrinV + schedDefaultPrinV + defaultV, period_with_lag)
        schedPrinV = pad_zeros(schedPrinV, period_with_lag)
        prepayPrinV = pad_zeros(prepayPrinV, period_with_lag)
        refundPrinV = pad_zeros(refundPrinV, period_with_lag)
        defaultV = pad_zeros(defaultV, period_with_lag)
        actInterestV = pad_zeros(actInterestV, period_with_lag)
        b_balanceV = pad_zeros(b_balanceV, period_with_lag)
        actualBalanceV = pad_zeros(actualBalanceV, period_with_lag)

        # Create scenario DataFrame
        df = pd.DataFrame({
            "Months": monthsV,
            "Prin": totalPrinV,
            "SchedPrin": schedPrinV,
            "Prepay Prin": prepayPrinV,
            "Refund Prin": refundPrinV,
            "Default": defaultV,
            "Writedown": writedownV,
            "Recovery": recoveryV,
            "Interest": actInterestV,
            "Servicing Fee": servicingFeeV,
            "Beginning Balance": b_balanceV,
            "Balance": actualBalanceV,
            "CFL": cfV,
            "Total Default": totalDefaultV,
        })
        
        return df
        
    def y2p(self, scenario, input, config): # yield to price
        yV = (1 + input.yieldValue/12)**np.arange(1, self.wam + 1 + scenario.recovery_lag) # len of months + recovery_lag

        cfV = self.getCashflow(scenario, config)["CFL"].values
        servicingFeeV = self.getCashflow(scenario, config)["Servicing Fee"].values
        refundPrinV = self.getCashflow(scenario, config).get("Refund Prin").values

        px = np.sum((cfV - servicingFeeV) / yV) / (self.pv - np.sum(refundPrinV / yV))
        return px
    

    def p2y(self, scenario, input, config): # price to yield
        price_target = input.fullpx

        def price_for_yield(y):
            class Helper:
                def __init__(self, yieldValue):
                    self.yieldValue = yieldValue
            input_obj = Helper(y)
            return self.y2p(scenario, input_obj, config) - price_target

        yield_solution = brentq(price_for_yield, 0.0001, 1.0)
        return yield_solution
