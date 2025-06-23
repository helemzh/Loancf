# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
from loancf import Loan, Input, Scenario, cpr2smm
import torch
from joblib import Parallel, delayed

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
    y = Input(yieldValue=.1)

    for att, val, px_expe in [('aggMDR', 0, 1.0986671204321417),
                              ('compIntHC', .4, 1.0961567650292567),
                              ('aggMDR', 0.04, 1.05029781840),
                              ('recovery_lag', 4, 1.05022636607),
                         ]:
        setattr(scenario, att, val)
        df = loan.getCashflow(scenario)
        px = loan.y2p(scenario, y)
        print(df, px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
def test_st2():
    """Short-dated loan"""
    wam = 12
    rate_redV = np.array([.0001] * 6 + [.0002] * 6)
    loan = Loan(wac=0.30, wam=wam, pv=100, rate_redV=rate_redV) # WAC is APR

    aggMDR_timingV = .01*np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1]) #loss timing
    assert np.isclose(aggMDR_timingV.sum(), 1.0, rtol=0, atol=1e-12)
    
    scenario = Scenario(
        smmV = cpr2smm(np.array([.35] * wam)), # curtailment CPR to SMM
        dqV=np.full(wam, 0),
        mdrV= 0,
        sevV=np.full(wam, 0.94),
        aggMDR=.03, #aggMDR is CGL
        aggMDR_timingV=aggMDR_timingV,
        refund_smm=cpr2smm(.01*np.array([74, 15, 5, 3, 2, 1] + [0]*(wam-6))), #refund cpr to smm   
        compIntHC= .2, # prepayment haircut,
        servicing_fee=0.02,
        servicing_fee_method='avg'
    )
    y = Input(yieldValue=0.1)

    for att, val, px_expe in [('recovery_lag', 4, 1.05063070588810)]:
        setattr(scenario, att, val)
        df = loan.getCashflow(scenario)
        px = loan.y2p(scenario, y)
        print(df,px)
        assert np.isclose(px, px_expe, rtol=0, atol=1e-12), \
            f'Actual v expected: {px} v {px_expe} for {att}={val}'
        
    # testing p2y
    p = Input(fullpx=1.05063070588810)
    y2 = loan.p2y(scenario, p)
    print("Yield:", y2)
    assert np.isclose(y2, 0.1, rtol=0, atol=1e-12)

def test_st3():
    n_loans = 200
    m_scenarios = 200
    wam = 12
    max_periods = wam

    loans = [Loan(wac=0.30 + 0.0001*i, wam=wam, pv=100 + 10*i) for i in range(n_loans)]

    # scenarios
    aggMDR_timingV = .01 * np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1])
    scenarios = []
    for j in range(m_scenarios):
        scenario = Scenario(
            smmV=cpr2smm(np.array([.35 + 0.0001*j] * wam)),
            dqV=np.full(wam, 0),
            mdrV=0,
            sevV=np.full(wam, 0.94),
            aggMDR=.03,
            aggMDR_timingV=aggMDR_timingV,
            refund_smm=cpr2smm(.01 * np.array([74, 15, 5, 3, 2, 1] + [0]*(wam-6))),
            compIntHC=.2,
            servicing_fee=0.02,
            servicing_fee_method='avg',
            # recovery_lag=4
        )
        scenarios.append(scenario)

    # Get column names from a sample DataFrame
    sample_df = loans[0].getCashflow(scenarios[0])
    columns = sample_df.columns
    n_features = len(columns)

    # Preallocate tensor: [n_loans, m_scenarios, max_periods, n_features]
    df_tensor = torch.zeros((n_loans, m_scenarios, max_periods, n_features), dtype=torch.float32)

    # Fill tensor with all DataFrame columns
    for i, loan in enumerate(loans):
        for j, scenario in enumerate(scenarios):
            df = loan.getCashflow(scenario)
            arr = df.to_numpy(dtype=np.float32)
            if arr.shape[0] < max_periods:
                pad = np.zeros((max_periods - arr.shape[0], n_features), dtype=np.float32) # can pad if needed
                arr = np.vstack([arr, pad])
            df_tensor[i, j, :, :] = torch.from_numpy(arr)

    torch.set_printoptions(linewidth=200)
    print("Tensor shape:", df_tensor.shape)  # (n_loans, m_scenarios, periods, n_features)
    print("Columns:", list(columns))
    # Example: access all columns for loan 0, scenario 0
    print("Loan 0, Scenario 0 all columns:\n", df_tensor[0, 0, :, :])
    print("Loan 0, Scenario 0 all columns:\n", df_tensor[49, 99, :, :])

    return df_tensor, columns

def test_st4():
    # parallel computation with joblib
    wac_arr = np.array([0.03, 0.04, 0.05])
    pv_arr = np.array([100000, 150000, 200000])
    wam_arr = np.array([12, 12, 12])

    smmV_arr = np.array([[0.01]*12, [0.02]*12])  # shape (2, 12)
    dqV_arr = np.zeros((2, 12))
    mdrV_arr = np.zeros((2, 12))
    sevV_arr = np.ones((2, 12)) * 0.2

    # Create Loan and Scenario objects
    loans = [Loan(wac=wac_arr[i], wam=wam_arr[i], pv=pv_arr[i]) for i in range(len(wac_arr))]
    scenarios = [
        Scenario(
            smmV=smmV_arr[j],
            dqV=dqV_arr[j],
            mdrV=mdrV_arr[j],
            sevV=sevV_arr[j],        
        )
        for j in range(smmV_arr.shape[0])
    ]

    # Prepare all pairs
    pairs = [(loan, scenario) for loan in loans for scenario in scenarios]
    n_loans = len(loans)
    n_scenarios = len(scenarios)

    # Get column info from a sample
    sample_df = loans[0].getCashflow(scenarios[0])
    columns = sample_df.columns
    n_features = len(columns)
    max_periods = sample_df.shape[0]

    # Parallel computation
    from joblib import Parallel, delayed
    def get_cf_arr(loan, scenario, max_periods, n_features):
        df = loan.getCashflow(scenario)
        arr = df.to_numpy(dtype=np.float32)
        if arr.shape[0] < max_periods:
            pad = np.zeros((max_periods - arr.shape[0], n_features), dtype=np.float32)
            arr = np.vstack([arr, pad])
        return arr

    results = Parallel(n_jobs=-1)(
        delayed(get_cf_arr)(loan, scenario, max_periods, n_features)
        for loan, scenario in pairs
    )

    df_tensor = torch.stack([torch.from_numpy(arr) for arr in results])
    df_tensor = df_tensor.view(n_loans, n_scenarios, max_periods, n_features)

    torch.set_printoptions(linewidth=200)
    print("Columns:", list(columns))
    print("Loan 0, Scenario 0 all columns:\n", df_tensor[0, 0, :, :])

    return df_tensor, columns

def test_st5():
    # create arrays for loans and scenarios
    loansV = np.array([
    [0.03, 12, 100000],
    [0.04, 12, 150000],
    [0.05, 12, 200000]
    ])
    # scenario arrays
    refund_smmV = np.array([
    cpr2smm(.01 * np.array([74, 15, 5, 3, 2, 1] + [0]*(12-6))),
    cpr2smm(.01 * np.array([50, 20, 10, 10, 5, 5] + [0]*(12-6))),
    cpr2smm(.01 * np.array([50, 20, 10, 10, 5, 5] + [0]*(12-6))),
    ])
    dqV = np.array([
    [np.zeros(12)],
    [np.zeros(12)],
    [np.zeros(12)]
    ])
    sevV = np.array([
    [np.full(12, 0.94)],
    [np.full(12, 0.95)],
    [np.full(12, 0.96)]
    ])
    smmV = np.array([
    [np.full(12, 0.35)],
    [np.full(12, 0.30)],
    [np.full(12, 0.40)]
    ])

    aggMDR_timingV = np.array([
    [0.01 * np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1])],
    [0.01 * np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1])],
    [0.01 * np.array([23, 10, 10, 10, 10, 10, 8, 7, 5, 4, 2, 1])]
    ])

    mdr = np.array([0,0,0])
    aggMDR = np.array([.03, .03, .03])
    compIntHC = np.array([.2, .2, .2])
    servicing_fee = np.array([0.02, 0.02, 0.02])
    servicing_fee_method = np.array(['avg', 'avg', 'avg'])


    # Loans
    loans = [Loan(wac=row[0], wam=int(row[1]), pv=row[2]) for row in loansV]

    # Scenarios
    scenarios = [
        Scenario(
            smmV=smmV[i][0],
            dqV=dqV[i][0],
            mdrV=mdr[i],
            sevV=sevV[i][0],
            aggMDR=aggMDR[i],
            aggMDR_timingV=aggMDR_timingV[i][0],
            refund_smm=refund_smmV[i],
            compIntHC=compIntHC[i],
            servicing_fee=servicing_fee[i],
            servicing_fee_method=servicing_fee_method[i],
            recovery_lag=0
        )
        for i in range(len(refund_smmV))
    ]

    n_loans = len(loans)
    m_scenarios = len(scenarios)
    sample_df = loans[0].getCashflow(scenarios[0])
    n_features = len(sample_df.columns)
    max_periods = sample_df.shape[0]

    results = []
    for loan in loans:
        for scenario in scenarios:
            df = loan.getCashflow(scenario)
            arr = df.to_numpy(dtype=np.float32)
            if arr.shape[0] < max_periods:
                pad = np.zeros((max_periods - arr.shape[0], n_features), dtype=np.float32)
                arr = np.vstack([arr, pad])
            results.append(arr)

    df_tensor = torch.stack([torch.from_numpy(arr) for arr in results])
    df_tensor = df_tensor.view(n_loans, m_scenarios, max_periods, n_features)

    torch.set_printoptions(linewidth=200)
    print("Tensor shape:", df_tensor.shape)
    print("Loan 0, Scenario 0 all columns:\n", df_tensor[0, 0, :, :])


if __name__ == '__main__':
    while(1):
        test_st2()
        break
    
        test_py1()
        test_py2()
        test_st()
