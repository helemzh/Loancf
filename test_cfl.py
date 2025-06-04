# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
from main import Loan, Yield, Output, Scenario, cpr2smm

def test_py1():
    loan = Loan(wac=0.0632, wam=357, pv=100000000)

    # Test cases: (smm, mdr, sev, yield, expected_px)
    test_cases = [
        (0.01, 0.1, 0.2, 0.055, 0.8281593283862828),
        (0, 0, 0, 0.055, 1.0919218749),
        (0, 0, 0, 0.0632, 1.0000000000),
        (0.01, 0, 0, 0.055, 1.0423018980),
        (0.01, 0, 0, 0.0632, 1.0000000000),
        (0.01, 0.1, 0, 0.055, 1.0013567128),
        (0.01, 0.1, 0, 0.0632, 0.9954659556),
        (0.01, 0.1, 0.2, 0.055, 0.8281593284),
        (0.01, 0.1, 0.2, 0.0632, 0.8232870530)
        # Add more test cases here
    ]

    for smm, mdr, sev, yield_value, expected_px in test_cases:
        recovery_lagValue = 0
        aggMDR_Value = 0.0
 
        aggMDR_timing_Vec = np.zeros(loan.wam, dtype=float)
             
        scenario = Scenario(smmV=smm, dqV=0, mdrV=mdr, sevV=sev, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)   
        
        # Create yield object
        y = Yield(yieldValue=yield_value)
        
        # Create output object
        output = Output(loan=loan, scenario=scenario, px=y)
        
        df = output.getCashflow()
        # Get price
        px = output.getPX()
        
        # Assert
        assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
            f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"

    print("All tests passed successfully!")

def test_py2():
    #test wal
    loan = Loan(wac=0.0632, wam=357, pv=100000000)

    # Test cases: (smm, mdr, sev, yield, expected_px)
    test_cases = [
     
        (0, 0, 0, 0.0632, 1.0000000000, 19.3142148090, 19.3142148090, 11.3140631626, 14.9166666667),
        (0.01, 0, 0, 0.0632, 1.0000000000, 6.9859506262, 6.9859506262, 6.2378547017, 6.7568208173),
        (0.01, 0.1, 0, 0.0632, 0.9954659556, 0.7514701165, 0.7514701165, 0.7511465462, 0.7514568528),
        (0.01, 0.1, 0.2, 0.0632, 0.8232870530, 0.7515413139, 0.7514701165, 0.7511465462, 0.7515217476)

        # Add more test cases here
    ]
    recovery_lagValue = 0
    aggMDR_Value = 0.0
 
    aggMDR_timing_Vec = np.zeros(loan.wam, dtype=float)
    
    for smm, mdr, sev, yield_value, expected_px, expected_wal_PrinV, expected_wal_BalanceDiffV, expected_wal_InterestV, expected_wal_cfl  in test_cases:
        dqV = mdr
        scenario = Scenario(smmV=smm, dqV=0, mdrV=mdr, sevV=sev, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)
        
        
        # Create yield object
        y = Yield(yieldValue=yield_value)
        
        # Create output object
        output = Output(loan=loan, scenario=scenario, px=y)
        
        df = output.getCashflow()
        # Get price
        px = output.getPX()

        wal_PrinV = output.get_wal_PrinV()
        wal_BalanceDiffV = output.get_wal_BalanceDiffV()
        wal_InterestV = output.get_wal_InterestV()
        wal_cfl = output.get_wal_cfl()  
        
        # Assert
        assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
            f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"
        
      
        assert np.isclose(wal_PrinV, expected_wal_PrinV, rtol=0, atol=1e-8)
        assert np.isclose(wal_BalanceDiffV, expected_wal_BalanceDiffV, rtol=0, atol=1e-8)
        assert np.isclose(wal_InterestV, expected_wal_InterestV, rtol=0, atol=1e-8)
        assert np.isclose(wal_cfl, expected_wal_cfl, rtol=0, atol=1e-8)
        
        # assert without wal
        # assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
        #    f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"

def test_st():
    """Short-dated loan"""
    wam = 12
    loan = Loan(wac=0.32, wam=wam, pv=1000000)
    aggMDR_timingV = .01*np.array([30,15,15,15,10,10,5]+[0]*(wam-7))
    assert np.isclose(aggMDR_timingV.sum(), 1.0, rtol=0, atol=1e-12)
    scenario = Scenario(
        smmV=np.full(wam, cpr2smm(.45)),
        dqV=np.full(wam, 0),
        mdrV=np.full(wam, 0),
        sevV=np.full(wam, .95),
        aggMDR=0, aggMDR_timingV=aggMDR_timingV,
        refund_smm=cpr2smm(np.array([.7, .2, .1, .05, .02, .01] + [0]*(wam-6))))
    y = Yield(yieldValue=.1)

    for att, val, px_expe in [('aggMDR', 0, 1.0986671204321417),
                              ('compIntHC', .4, 1.0961567650292567),
                              ('aggMDR', 0.04, 1.05029781840),
                              ('recovery_lag', 4, 1.05022636607),
                         ]:
        setattr(scenario, att, val)
        output = Output(loan=loan, scenario=scenario, px=y)
        df = output.getCashflow()
        px = output.getPX()
        print(df, px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
def test_st2():
    """Short-dated loan"""
    wam = 12
    loan = Loan(wac=0.30, wam=wam, pv=100) # WAC is APR
    aggMDR_timingV = .01*np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1]) #loss timing
    assert np.isclose(aggMDR_timingV.sum(), 1.0, rtol=0, atol=1e-12)
    
    scenario = Scenario(
        smmV = cpr2smm(np.array([.35] * wam)), # curtailment CPR to SMM
        dqV=np.full(wam, 0),
        mdrV= 0,
        sevV=np.full(wam, 0.94),
        aggMDR=.03, aggMDR_timingV=aggMDR_timingV, #aggMDR is CGL
        refund_smm=cpr2smm(.01*np.array([74, 15, 5, 3, 2, 1] + [0]*(wam-6))), #refund cpr to smm   
        compIntHC= .2, # prepayment haircut,
        servicing_fee=0
    )
    y = Yield(yieldValue=0.1)

    for att, val, px_expe in [
                              ('recovery_lag', 4, 1.05944025156667),
                         ]:
        setattr(scenario, att, val)
        output = Output(loan=loan, scenario=scenario, px=y)
        df = output.getCashflow()
        px = output.getPX()
        print(df, px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'


if __name__ == '__main__':
    while(1):
        test_st2()
        # test_st()
        break
    
        test_py1()
        test_py2()
        test_st()
