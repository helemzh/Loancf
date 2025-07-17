# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
from loancf import Loan, Input, Scenario, Config, cpr2smm, calc, y2bey, bey2y

def test_py1():
    '''
    Testing different scenarios
    Test cases: (smm, mdr, sev, yield, expected_px)
    '''
    wam = 357
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
        loan = Loan(wac=0.0632, wam=wam, pv=100000000)
        y = Input(yieldValue=yield_value)
        config = Config() 
        scenario = Scenario(
            smmV=np.full(wam, smm), 
            dqV=np.full(wam, 0), 
            mdrV=np.full(wam, mdr), 
            sevV=np.full(wam, sev),
        )

        df = loan.getCashflow(scenario, config)
        px = loan.y2p(scenario, y, config)
        
        # Assert
        assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
            f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"

    print("All tests passed successfully!")

def test_py2():
    '''
    Testing for WAL
    Test cases: (smm, mdr, sev, yield, expected_px)
    '''
    test_cases = [
        (0, 0, 0, 0.0632, 1.0000000000, 19.3142148090, 19.3142148090, 11.3140631626, 14.9166666667),
        (0.01, 0, 0, 0.0632, 1.0000000000, 6.9859506262, 6.9859506262, 6.2378547017, 6.7568208173),
        (0.01, 0.1, 0, 0.0632, 0.9954659556, 0.7514701165, 0.7514701165, 0.7511465462, 0.7514568528),
        (0.01, 0.1, 0.2, 0.0632, 0.8232870530, 0.7515413139, 0.7514701165, 0.7511465462, 0.7515217476)
    ]
    wam = 357
    for smm, mdr, sev, yield_value, expected_px, expected_wal_PrinV, expected_wal_BalanceDiffV, expected_wal_InterestV, expected_wal_cfl  in test_cases:
        loan = Loan(wac=0.0632, wam=wam, pv=100000000)
        y = Input(yieldValue=yield_value)
        config = Config()
        scenario = Scenario(
            smmV=np.full(wam, smm), 
            dqV=np.full(wam, 0), 
            mdrV=np.full(wam, mdr), 
            sevV=np.full(wam, sev),
            )
        
        df = loan.getCashflow(scenario, config)
        px = loan.y2p(scenario, y, config)


        wal_PrinV = calc(df["Prin"])
        wal_BalanceDiffV = calc(df["Beginning Balance"] - df["Balance"])
        wal_InterestV = calc(df["Interest"])
        wal_cfl = calc(df["CFL"]) 
        
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
    y = Input(yieldValue=.1)
    config = Config()

    for att, val, px_expe in [('aggMDR', 0, 1.0986671204321417),
                              ('compIntHC', .4, 1.0961567650292567),
                              ('aggMDR', 0.04, 1.05029781840),
                              ('recovery_lag', 4, 1.05022636607),
                         ]:
        setattr(scenario, att, val)
        df = loan.getCashflow(scenario, config)
        px = loan.y2p(scenario, y, config)
        print(df, px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
def test_st2():
    """
    p2y
    Short-dated loan
    """
    wam = 12
    loan = Loan(wac=0.30, wam=wam, pv=100) # WAC is APR

    aggMDR_timingV = .01*np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1]) #loss timing
    assert np.isclose(aggMDR_timingV.sum(), 1.0, rtol=0, atol=1e-12)
    
    scenario = Scenario(
        rate_redV = np.array([.000] * wam),
        smmV = cpr2smm(np.array([.35] * wam)), # curtailment CPR to SMM
        dqV=np.full(wam, 0),
        mdrV= 0,
        sevV=np.full(wam, 0.94),
        aggMDR=.03, #aggMDR is CGL
        aggMDR_timingV=aggMDR_timingV,
        refund_smm=cpr2smm(.01*np.array([74, 15, 5, 3, 2, 1] + [0]*(wam-6))), #refund cpr to smm   
        compIntHC= .2, # prepayment haircut,
        servicing_fee=0.02,
    )
    y = Input(yieldValue=0.1)
    config = Config()

    for att, val, px_expe in [('recovery_lag', 4, 1.05063070588810)]:
        setattr(scenario, att, val)
        df = loan.getCashflow(scenario, config)
        px = loan.y2p(scenario, y, config)
        print(df,px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
    # testing p2y
    p = Input(fullpx=1.05063070588810)
    y2 = loan.p2y(scenario, p, config)
    print("Yield:", y2)
    assert np.isclose(y2, 0.1, rtol=0, atol=1e-12)

def test_st3():
    """
    0 Scenario, unfixed rate
    Short-dated loan
    """
    wam = 12
    loan = Loan(wac=0.30, wam=wam, pv=100) # WAC is APR
    
    scenario = Scenario(
        smmV = cpr2smm(np.array([0] * wam)), # curtailment CPR to SMM
        dqV=np.full(wam, 0),
        mdrV= 0,
        sevV=np.full(wam, 0),
        aggMDR=0, #aggMDR is CGL
        compIntHC= 0, # prepayment haircut,
        servicing_fee=0,
        rate_redV= np.array([0.0001] * wam)
    )
    y = Input(yieldValue=0.1)
    config = Config(rate_red_method=True)

    for att, val, px_expe in [('recovery_lag', 0, 1.10881219814726)]:
        setattr(scenario, att, val)
        df = loan.getCashflow(scenario, config)
        px = loan.y2p(scenario, y, config)
        print(df,px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
def test_dqadvance():
    """
    Intex test case for dq_adv_int and dq_adv_prin 
    """
    wam = 72
    wac = 0.1585
    pv = 9_498_315.68
    loan = Loan(wac=wac, wam=wam, pv=pv) # WAC is APR
    
    scenario = Scenario(
        smmV = cpr2smm(np.full(wam, .12)),
        dqV= np.full(wam, 0),
        mdrV= cpr2smm(np.array([0, 0, 0, .1] + [0.1] * (wam-4))),
        sevV= np.full(wam, 1),
        aggMDR= 0,
        compIntHC= 0,
        servicing_fee=0,
        #rate_redV= np.array([0.0001] * wam),
        dq_adv_prin=0,
        dq_adv_int=0,
        is_advance=True
    )
    input = Input(yieldValue=bey2y(0.1))
    config = Config(rate_red_method=False)

    df = loan.getCashflow(scenario, config)
    px = loan.y2p(scenario, input, config)
    assert np.isclose(px, .9381589717, rtol=0, atol=1e-8)

    # with is_advance = False
    scenario.is_advance = False
    df = loan.getCashflow(scenario, config)
    px = loan.y2p(scenario, input, config)
    assert np.isclose(px, .93577210138, rtol=0, atol=1e-8)

    # with dqV
    scenario.dqV = .01 * np.array([1, 2, 3, 10] + [10] * (wam-4))
    scenario.is_advance = False
    df = loan.getCashflow(scenario, config)
    px = loan.y2p(scenario, input, config)
    assert np.isclose(px, .87181853008, rtol=0, atol=1e-10)

    # with dq_adv_prin & int
    scenario.dq_adv_prin = 0.7
    scenario.dq_adv_int = 0.3
    df = loan.getCashflow(scenario, config)
    px = loan.y2p(scenario, input, config)
    assert np.isclose(px, .90545638658, rtol=0, atol=1e-10)

    settle_day = 14
    accrued_interest = wac * (settle_day - 1) / 360

    print(df,px)      


if __name__ == '__main__':
    while(1):
        test_dqadvance()
        break
    