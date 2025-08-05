# Copyright (c) 2025 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
import torch
from scipy.optimize import newton
import tensor_loancf as tlcf
import loancf as lcf
import new_tensor_loancf as ntlcf


"""
Test cases for tensor_loancf
"""
def test_compare2d_3d():
    '''
    Test loancf and tensor_loancf by aggregating balances and comparing tensor and numpy results
    '''
    loans_tensor = torch.tensor([
        [0.3, 12, 100_000_000],
        [0.25, 13, 50_000_000],
        [0.20, 14, 30_000_000],
    ], dtype=torch.float64)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_loans = loans_tensor.shape[0]
    n_scenarios = 2
    scenarios_tensor = torch.zeros(n_scenarios, 14, max_wam)

    ### Scenario 0 ###
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :] = 0
    scenarios_tensor[0, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[0, tlcf.SMM_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_I, :] = 0
    scenarios_tensor[0, tlcf.MDR_I, :] = 0
    scenarios_tensor[0, tlcf.SEV_I, :] = 0
    scenarios_tensor[0, tlcf.AGGMDR_I, 0] = 0
    scenarios_tensor[0, tlcf.COMPINTHC_I, 0] = 0
    scenarios_tensor[0, tlcf.SERVICING_FEE_I, 0] = 0
    scenarios_tensor[0, tlcf.RECOVERY_LAG_I, 0] = 0
    scenarios_tensor[0, tlcf.REFUND_PREMIUM_I, 0] = 1.0
    scenarios_tensor[0, tlcf.RATE_RED_I, :] = 0.00015
    scenarios_tensor[0, tlcf.DQ_ADV_PRIN_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_ADV_INT_I, :] = 0
    ### Scenario 1 ###
    scenarios_tensor[1, tlcf.REFUND_SMM_I, :] = 0
    scenarios_tensor[1, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[1, tlcf.SMM_I, :] = 0.1
    scenarios_tensor[1, tlcf.DQ_I, :] = 0
    scenarios_tensor[1, tlcf.MDR_I, :] = 0
    scenarios_tensor[1, tlcf.SEV_I, :] = 0
    scenarios_tensor[1, tlcf.AGGMDR_I, 0] = 0
    scenarios_tensor[1, tlcf.COMPINTHC_I, 0] = 0
    scenarios_tensor[1, tlcf.SERVICING_FEE_I, 0] = 0
    scenarios_tensor[1, tlcf.RECOVERY_LAG_I, 0] = 0
    scenarios_tensor[1, tlcf.REFUND_PREMIUM_I, 0] = 1.0
    scenarios_tensor[1, tlcf.RATE_RED_I, :] = .0001
    scenarios_tensor[1, tlcf.DQ_ADV_PRIN_I, :] = 0
    scenarios_tensor[1, tlcf.DQ_ADV_INT_I, :] = 0

    # list of result dataframes from loancf
    dfs = []
    numpy_balances = []  # aggregated balance for each scenario
    
    # loop call loancf for each loan call each scenario
    for li in range(n_loans): #scenario index
        wam = int(loans_tensor[li, tlcf.L_WAM_I].item())
        loan = lcf.Loan(
            wac=loans_tensor[li, tlcf.L_WAC_I].cpu().numpy(), 
            wam=int(loans_tensor[li, tlcf.L_WAM_I]), 
            pv=loans_tensor[li, tlcf.L_PV_I].cpu().numpy()
        )
        
        for si in range(n_scenarios): #scenario index
            #generate cashflow for each index
            scenario = lcf.Scenario(
                smmV = scenarios_tensor[si, tlcf.SMM_I, :wam].cpu().numpy(),
                dqV = scenarios_tensor[si, tlcf.DQ_I, :wam].cpu().numpy(),
                mdrV = scenarios_tensor[si, tlcf.MDR_I, :wam].cpu().numpy(),
                sevV = scenarios_tensor[si, tlcf.SEV_I, :wam].cpu().numpy(),
                aggMDR = scenarios_tensor[si, tlcf.AGGMDR_I, :wam].cpu().numpy(),
                compIntHC = scenarios_tensor[si, tlcf.COMPINTHC_I, :wam].cpu().numpy(),
                servicing_fee = scenarios_tensor[si, tlcf.SERVICING_FEE_I, :wam].cpu().numpy(),
                rate_redV = scenarios_tensor[si, tlcf.RATE_RED_I, :wam].cpu().numpy()
            )

            config = lcf.Config(rate_red_method=True)
            df = loan.getCashflow(scenario, config)
            dfs.append(df)

    # retrieve numpy results
    numpy_balances = np.zeros((n_scenarios, max_wam))
    for li in range(n_loans):
        wam = int(loans_tensor[li, tlcf.L_WAM_I].item())
        for si in range(n_scenarios):
            idx = li * n_scenarios + si
            bbal = dfs[idx]["Beginning Balance"]
            if len(bbal) < max_wam:
                bbal = np.pad(bbal, (0, max_wam - len(bbal)))
            numpy_balances[si] += bbal
    numpy_balances_total = numpy_balances.sum(axis=0, keepdims=True)  # shape: [1, max_wam]

    # retrieve tensor results
    model = tlcf.LoanAmort(loans_tensor)
    config = tlcf.Config(rate_red_method=True)
    result_tensor, feature_names = model(scenarios_tensor, config)
    tensor_aggbalances = result_tensor[:,:,:, tlcf.RESULT_BEGINNING_BALANCE_I].sum(dim=(0,1)) #aggregate tensor results
    assert np.allclose(numpy_balances_total, tensor_aggbalances.cpu().numpy()), "Scenario vector balances do not match"

def test_tensor_aggyieldprice():
    # Example: 1 loan, 2 scenario
    # loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
    loans_tensor = torch.tensor([
        [0.3, 12, 100],
    ], dtype=torch.float64)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_loans = loans_tensor.shape[0]
    n_scenarios = 2
    scenarios_tensor = torch.zeros(n_scenarios, 14, max_wam, dtype=torch.float64)

    ### Scenario 0 ###
    # Set up refund_smm and aggMDR_timingV
    refund_smm = tlcf.cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1], dtype=torch.float64)) # input refund_smm values
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :] = 0 # first fill 0
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :len(refund_smm)] = refund_smm
    aggMDR_timing = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1], dtype=torch.float64) # input aggMDR_timing values
    scenarios_tensor[0, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[0, tlcf.AGGMDR_TIMING_I, :len(aggMDR_timing)] = aggMDR_timing
    scenarios_tensor[0, tlcf.SMM_I, :] = tlcf.cpr2smm(0.35)
    scenarios_tensor[0, tlcf.DQ_I, :] = 0
    scenarios_tensor[0, tlcf.MDR_I, :] = 0
    scenarios_tensor[0, tlcf.SEV_I, :] = 0.94
    scenarios_tensor[0, tlcf.AGGMDR_I, 0] = 0.03
    scenarios_tensor[0, tlcf.COMPINTHC_I, 0] = 0.2
    scenarios_tensor[0, tlcf.SERVICING_FEE_I, 0] = 0.02
    scenarios_tensor[0, tlcf.RECOVERY_LAG_I, 0] = 4
    scenarios_tensor[0, tlcf.REFUND_PREMIUM_I, 0] = 1.0
    scenarios_tensor[0, tlcf.RATE_RED_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_ADV_PRIN_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_ADV_INT_I, :] = 0
    ### Scenario 1 ###
    scenarios_tensor[1, tlcf.REFUND_SMM_I, :] = 0
    scenarios_tensor[1, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[1, tlcf.SMM_I, :] = tlcf.cpr2smm(0)
    scenarios_tensor[1, tlcf.DQ_I, :] = 0
    scenarios_tensor[1, tlcf.MDR_I, :] = 0
    scenarios_tensor[1, tlcf.SEV_I, :] = 0
    scenarios_tensor[1, tlcf.AGGMDR_I, 0] = 0
    scenarios_tensor[1, tlcf.COMPINTHC_I, 0] = 0
    scenarios_tensor[1, tlcf.SERVICING_FEE_I, 0] = 0
    scenarios_tensor[1, tlcf.RECOVERY_LAG_I, 0] = 0
    scenarios_tensor[1, tlcf.REFUND_PREMIUM_I, 0] = 1.0
    scenarios_tensor[1, tlcf.RATE_RED_I, :] = .0001
    scenarios_tensor[1, tlcf.DQ_ADV_PRIN_I, :] = 0
    scenarios_tensor[1, tlcf.DQ_ADV_INT_I, :] = 0

    model = tlcf.LoanAmort(loans_tensor)
    config = tlcf.Config(rate_red_method=True)
    result_tensor, feature_names = model(scenarios_tensor, config)

    print("Result shape:", result_tensor.shape)  # [n_loans, n_scenarios, max_wam+lag, n_features]
    # print(f"Result for loan 0, scenario 0:\n\t{feature_names}\n{result_tensor[0, 0]}")

    # Price and Yield for each loan/scenario
    yield_input = torch.full((n_loans, n_scenarios), 0.1, device=loans_tensor.device, dtype=torch.float64)
    price_result = tlcf.y2p_tensor(loans_tensor, scenarios_tensor, yield_input, config)

    price_input = torch.tensor([[1.0506307058881, 1.10881219814726]], dtype=torch.float64)
    yield_result = tlcf.p2y_tensor(loans_tensor, scenarios_tensor, price_input, config)

    print(f"price{price_result}, \nyield{yield_result}")
    assert torch.allclose(price_result, price_input, rtol=0, atol=1e-12), "Price calculation mismatch"
    assert torch.allclose(yield_result, yield_input, rtol=0, atol=1e-12), "Yield calculation mismatch"

    '''    # Example yield-price aggregation, when loan method: set tensor size as (n_loans,) pool method: (n_scenarios,)
    yield_tensor = torch.full((n_scenarios,), 0.10, device=loans_tensor.device)
    price_tensor = tlcf.yield_price_aggregation(
        loans_tensor, scenarios_tensor, yield_tensor, function='y2p', method='pool'
    )
    print(f"Aggregated price: {price_tensor}")


    target_price = torch.full((n_scenarios,), 1.0506307058881, device=loans_tensor.device)
    yield_tensor = tlcf.yield_price_aggregation(
        loans_tensor, scenarios_tensor, target_price, function='p2y', method='pool'
    )   
    print(f"Aggregated yield: {yield_tensor}")
    '''

def test_tensor_dqadvance():
    # Example: 1 loan, 3 scenario
    # loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
    loans_tensor = torch.tensor([
        [0.1585, 72, 9_498_315.68],
    ], dtype=torch.float64)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_loans = loans_tensor.shape[0]
    n_scenarios = 3
    scenarios_tensor = torch.zeros(n_scenarios, 14, max_wam, dtype=torch.float64)

    ### Scenario 0 ###
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :] = 0
    scenarios_tensor[0, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[0, tlcf.SMM_I, :] = tlcf.cpr2smm(0.12)
    scenarios_tensor[0, tlcf.DQ_I, :] = 0
    
    mdrV = tlcf.cpr2smm(torch.tensor([0, 0, 0,], dtype=torch.float64)) # input refund_smm values
    scenarios_tensor[0, tlcf.MDR_I, :] = tlcf.cpr2smm(0.1) # first fill 0
    scenarios_tensor[0, tlcf.MDR_I, :len(mdrV)] = mdrV

    scenarios_tensor[0, tlcf.SEV_I, :] = 1
    scenarios_tensor[0, tlcf.AGGMDR_I, 0] = 0
    scenarios_tensor[0, tlcf.COMPINTHC_I, 0] = 0
    scenarios_tensor[0, tlcf.SERVICING_FEE_I, 0] = 0
    scenarios_tensor[0, tlcf.RECOVERY_LAG_I, 0] = 0
    scenarios_tensor[0, tlcf.REFUND_PREMIUM_I, 0] = 0
    scenarios_tensor[0, tlcf.RATE_RED_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_ADV_PRIN_I, :] = 0
    scenarios_tensor[0, tlcf.DQ_ADV_INT_I, :] = 0

    ### Scenario 1: dqV ###
    scenarios_tensor[1, :, :] = scenarios_tensor[0, :, :] # copy scenario 0
    dqV = .01 * torch.tensor([1, 2, 3], dtype=torch.float64)
    scenarios_tensor[1, tlcf.DQ_I, :] = .1
    scenarios_tensor[1, tlcf.DQ_I, :len(dqV)] = dqV

    ### Scenario 2: dq adv prin and int ###
    scenarios_tensor[2, :, :] = scenarios_tensor[1, :, :] # copy scenario 1
    scenarios_tensor[2, tlcf.DQ_ADV_PRIN_I, :] = 0.7
    scenarios_tensor[2, tlcf.DQ_ADV_INT_I, :] = 0.3

    model = tlcf.LoanAmort(loans_tensor)
    config = tlcf.Config(is_advance=True)
    result_tensor, feature_names = model(scenarios_tensor, config)
    print("Result shape:", result_tensor.shape)  # [n_loans, n_scenarios, max_wam+lag, n_features]

    # Price and Yield for each loan/scenario
    yield_input = torch.full((n_loans, n_scenarios), tlcf.bey2y(0.1), device=loans_tensor.device)
    price_result = tlcf.y2p_tensor(loans_tensor, scenarios_tensor, yield_input, config)
    price_input = torch.tensor([[0.9381589717, 0.9020295044, 0.92732012833]], dtype=torch.float64)

    assert torch.allclose(price_result, price_input, rtol=0, atol=1e-8), "Advanced Price calculation mismatch"
    print("Advanced Price:", price_result)

    config_not_advance = tlcf.Config(is_advance=False)
    price_result = tlcf.y2p_tensor(loans_tensor, scenarios_tensor, yield_input, config_not_advance)
    price_input = torch.tensor([[0.93577210138, 0.87181853008, 0.90545638658]], dtype=torch.float64)

    assert torch.allclose(price_result, price_input, rtol=0, atol=1e-8), "Non-advanced Price calculation mismatch"
    print("Non-advanced Price:", price_result)

"""
Test cases for new_tensor_loancf (restructured [N, WL, M] result tensors)
"""
def test_ntlcf():
    # Initial testing for ntlcf
    N = 2 # n loans
    M = 2 # m scenarios
    W = 3 # max wam

    orig_wacV = torch.tensor([[0.06], [0.05]]).repeat(1, W)  # [N, W]
    wam = torch.tensor([[2], [3]])  # [N, 1]
    pv = torch.tensor([[100.], [200.]])  # [N, 1]
    rate_redV = (torch.tensor([.0001, .0002]) + (torch.arange(W).unsqueeze(1) * .0001))  # [W, M]

    refund_smmV = torch.zeros((W, M))  # [W, M]
    smmV = ntlcf.cpr2smm(torch.tensor([[.35, 0]]).repeat(W, 1))  # [W, M]
    dqV = torch.tensor([[.001, 0]]).repeat(W, 1)  # [W, M]
    mdrV = torch.tensor([[.03, 0]]).repeat(W, 1)  # [W, M]
    sevV = torch.vstack([torch.tensor([.95, .96, .97]), torch.ones(W)]).t()  # [W, M]
    aggMDR_timingV = torch.tensor([[.5, .3, .1], [.5, .3, .1]]).t()  # [W, M]

    aggMDR = torch.tensor([[.03, 0]])  # [1, M]
    compIntHC = torch.tensor([[.2, .3]])  # [1, M]
    servicing_fee = torch.tensor([[.02, 0.01]])  # [1, M]
    recovery_lag = torch.tensor([[4, 0]])  # [1, M]
    refund_premium = torch.full((1, M), 1)  # [1, M]
    dq_adv_prin = torch.full((1, M), 0)  # [1, M]
    dq_adv_int = torch.full((1, M), 0)  # [1, M]

    config = ntlcf.Config(agg_cf=True)
    model = ntlcf.LoanAmort()
    # Results in Class of tensors, all and aggregated
    result_tensor, agg_tensor = model(
        config=config, orig_wacV=orig_wacV, wam=wam, pv=pv, rate_redV=rate_redV,
        refund_smmV=refund_smmV, smmV=smmV, dqV=dqV, mdrV=mdrV, sevV=sevV,
        aggMDR_timingV=aggMDR_timingV, aggMDR=aggMDR,
        compIntHC=compIntHC, servicing_fee=servicing_fee,
        recovery_lag=recovery_lag, refund_premium=refund_premium,
        dq_adv_prin=dq_adv_prin, dq_adv_int=dq_adv_int
    )

    # Aggregated price/yield
    aggcalc = ntlcf.Calc(agg_tensor, pv, config)
    aggyld_input = torch.full((agg_tensor.cfV.shape[0], agg_tensor.cfV.shape[1]), 0.1)
    aggpx_result = aggcalc.y2p(aggyld_input)
    aggpx_input = torch.tensor([[[0.914025891997, 0.991725183301]]])
    assert torch.allclose(aggpx_result, aggpx_input, rtol=0, atol=1e-8), "Price mismatched"

    # Non-aggregated price/yield
    calc = ntlcf.Calc(result_tensor, pv, config)
    yld_input = torch.full((result_tensor.cfV.shape[0], result_tensor.cfV.shape[1]), 0.1)
    px_result = calc.y2p(yld_input)
    px_input = torch.tensor([[[0.926778247166, 0.994196293031]],
                              [[0.907649714412, 0.990489628436]]])
    assert torch.allclose(px_result, px_input, rtol=0, atol=1e-8), "Price mismatched"

    config2 = ntlcf.Config(mode="matched")
    result_tensor2 = model(
        config=config2, orig_wacV=orig_wacV, wam=wam, pv=pv, rate_redV=rate_redV,
        refund_smmV=refund_smmV, smmV=smmV, dqV=dqV, mdrV=mdrV, sevV=sevV,
        aggMDR_timingV=aggMDR_timingV, aggMDR=aggMDR,
        compIntHC=compIntHC, servicing_fee=servicing_fee,
        recovery_lag=recovery_lag, refund_premium=refund_premium,
        dq_adv_prin=dq_adv_prin, dq_adv_int=dq_adv_int
    )
    calc = ntlcf.Calc(result_tensor2, pv, config2)
    yld_input = torch.full((result_tensor2.cfV.shape[0], result_tensor2.cfV.shape[1]), 0.1)
    px_result = calc.y2p(yld_input)
    px_input = torch.tensor([[[0.926778247166]], [[0.990489628436]]])
    assert torch.allclose(px_result, px_input, rtol=0, atol=1e-8), "Price mismatched"
    
def test_ntlcf_aggyieldprice():
    N = 1 # n loans
    M = 2 # m scenarios
    W = 12 # max wam

    orig_wacV = torch.tensor([[0.3]]).repeat(N, W)           # [N, W]
    wam = torch.tensor([[12]])                               # [N, 1]
    pv = torch.tensor([[100.]])                              # [N, 1]
    rate_redV = torch.tensor([[0, 0.0001]]).repeat(W, 1)     # [W, M]

    refund_smmV = torch.full((W, M), 0, dtype=torch.float64) # [W, M]
    refund_smmV[:6, 0] = ntlcf.cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1]))

    smmV = ntlcf.cpr2smm(torch.tensor([[.35, 0]]).repeat(W, 1)) # [W, M]
    dqV = torch.zeros((W, M))                                # [W, M]
    mdrV = torch.zeros((W, M))                               # [W, M]
    sevV = torch.tensor([[0.94, 0.]]).repeat(W, 1)           # [W, M]

    aggMDR_timingV = torch.zeros((W, M))                     # [W, M]
    aggMDR_timingV[:12, 0] = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1])

    aggMDR = torch.tensor([[0.03, 0.0]])                     # [1, M]
    compIntHC = torch.tensor([[0.2, 0.0]])                   # [1, M]
    servicing_fee = torch.tensor([[0.02, 0.0]])              # [1, M]
    recovery_lag = torch.tensor([[4.0, 0.0]])                # [1, M]
    refund_premium = torch.ones(1, M)                        # [1, M]
    dq_adv_prin = torch.zeros(1, M)                          # [1, M]
    dq_adv_int = torch.zeros(1, M)                           # [1, M]

    config = ntlcf.Config(agg_cf=False, is_advance=False)
    model = ntlcf.LoanAmort()
    # Results in Class of tensors, all and aggregated
    result_advance = model(config=config, orig_wacV=orig_wacV, wam=wam, pv=pv, rate_redV=rate_redV, 
                   refund_smmV=refund_smmV, smmV=smmV, dqV=dqV, mdrV=mdrV, sevV=sevV,
                   aggMDR_timingV=aggMDR_timingV, aggMDR=aggMDR,
                   compIntHC=compIntHC, servicing_fee=servicing_fee,
                   recovery_lag=recovery_lag, refund_premium=refund_premium,
                   dq_adv_prin=dq_adv_prin, dq_adv_int=dq_adv_int)    

    # Advanced Scenarios
    calc = ntlcf.Calc(result_advance, pv, config)
    yld_input = torch.full((result_advance.cfV.shape[0], result_advance.cfV.shape[1]), 0.1)
    px_result = calc.y2p(yld_input)
    px_input = torch.tensor([[[1.0506307058881, 1.10881219814726]]])
    assert torch.allclose(px_result, px_input, rtol=0, atol=1e-8), "Price mismatched"
    print(f" Price: {px_result}")

def test_ntlcf_dqadvance():
    N = 1 # n loans
    M = 3 # m scenarios
    W = 72 # max wam

    orig_wacV = torch.tensor([[0.1585]]).repeat(N, W)         # [N, W]
    wam = torch.tensor([[72]])                                # [N, 1]
    pv = torch.tensor([[9_498_315.68]])                       # [N, 1]
    rate_redV = torch.zeros((W, M))                           # [W, M]

    refund_smmV = torch.zeros((W, M))                         # [W, M]
    smmV = ntlcf.cpr2smm(torch.full((W, M), 0.12))            # [W, M]

    dqV_0 = torch.full((1, W), 0)                             # [1, W]
    dqV_1_2 = torch.full((2, W), 0.1)                         # [2, W]
    dqV_1_2[:, :3] = .01 * torch.tensor([1, 2, 3])            # [2, W]
    dqV = torch.cat([dqV_0, dqV_1_2], dim=0).t()              # [W, M]

    mdrV = torch.full((W, M), ntlcf.cpr2smm(0.1))             # [W, M]
    mdrV[:3, :] = 0
    sevV = torch.ones((W, M))                                 # [W, M]
    aggMDR_timingV = torch.zeros((W, M))                      # [W, M]

    aggMDR = torch.zeros((1, M))                              # [1, M]
    compIntHC = torch.zeros((1, M))                           # [1, M]
    servicing_fee = torch.zeros((1, M))                       # [1, M]
    recovery_lag = torch.zeros((1, M))                        # [1, M]
    refund_premium = torch.zeros((1, M))                      # [1, M]
    dq_adv_prin = torch.tensor([[0, 0, 0.7]])                 # [1, M]
    dq_adv_int = torch.tensor([[0, 0, 0.3]])                  # [1, M]

    config = ntlcf.Config(agg_cf=False, is_advance=True)
    model = ntlcf.LoanAmort()
    # Results in Class of tensors, all and aggregated
    result_advance = model(config=config, orig_wacV=orig_wacV, wam=wam, pv=pv, rate_redV=rate_redV, 
                   refund_smmV=refund_smmV, smmV=smmV, dqV=dqV, mdrV=mdrV, sevV=sevV,
                   aggMDR_timingV=aggMDR_timingV, aggMDR=aggMDR,
                   compIntHC=compIntHC, servicing_fee=servicing_fee,
                   recovery_lag=recovery_lag, refund_premium=refund_premium,
                   dq_adv_prin=dq_adv_prin, dq_adv_int=dq_adv_int)    

    # Advanced Scenarios
    calc = ntlcf.Calc(result_advance, pv, config)
    yld_input = torch.full((result_advance.cfV.shape[0], result_advance.cfV.shape[1]), ntlcf.bey2y(0.1))
    px_result = calc.y2p(yld_input)
    px_input = torch.tensor([[[0.9381589717, 0.9020295044, 0.92732012833]]])
    assert torch.allclose(px_result, px_input, rtol=0, atol=1e-8), "Advanced Price mismatched"
    print(f"Advanced Price: {px_result}")

    # Non-advanced Scenarios
    config2 = ntlcf.Config(agg_cf=False, is_advance=False)
    result_nonadvance = model(config=config2, orig_wacV=orig_wacV, wam=wam, pv=pv, rate_redV=rate_redV, 
                   refund_smmV=refund_smmV, smmV=smmV, dqV=dqV, mdrV=mdrV, sevV=sevV,
                   aggMDR_timingV=aggMDR_timingV, aggMDR=aggMDR,
                   compIntHC=compIntHC, servicing_fee=servicing_fee,
                   recovery_lag=recovery_lag, refund_premium=refund_premium,
                   dq_adv_prin=dq_adv_prin, dq_adv_int=dq_adv_int)    
    calc = ntlcf.Calc(result_nonadvance, pv, config2)
    yld_input = torch.full((result_nonadvance.cfV.shape[0], result_nonadvance.cfV.shape[1]), ntlcf.bey2y(0.1))
    px_result = calc.y2p(yld_input)
    px_input = torch.tensor([[[0.93577210138, 0.87181853008, 0.90545638658]]])
    assert torch.allclose(px_result, px_input, rtol=0, atol=1e-8), "Non-advanced Price mismatched"
    print(f"Non-advanced Price: {px_result}")


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    test_ntlcf()