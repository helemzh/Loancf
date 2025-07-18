import pytest
import numpy as np
import torch
from scipy.optimize import newton
import tensor_loancf as tlcf
import loancf as lcf
import new_tensor_loancf as ntlcf


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
    scenarios_tensor = torch.zeros(n_scenarios, 14, max_wam)

    ### Scenario 0 ###
    # Set up refund_smm and aggMDR_timingV
    refund_smm = tlcf.cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1])) # input refund_smm values
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :] = 0 # first fill 0
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :len(refund_smm)] = refund_smm
    aggMDR_timing = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1]) # input aggMDR_timing values
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
    yield_input = torch.full((n_loans, n_scenarios), 0.1, device=loans_tensor.device)
    price_result = tlcf.y2p_tensor(loans_tensor, scenarios_tensor, yield_input, config)

    price_input = torch.tensor([[1.0506307058881, 1.10881219814726]])
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
    # Example: 1 loan, 2 scenario
    # loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
    loans_tensor = torch.tensor([
        [0.1585, 72, 9_498_315.68],
    ], dtype=torch.float64)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_loans = loans_tensor.shape[0]
    n_scenarios = 3
    scenarios_tensor = torch.zeros(n_scenarios, 14, max_wam)

    ### Scenario 0 ###
    scenarios_tensor[0, tlcf.REFUND_SMM_I, :] = 0
    scenarios_tensor[0, tlcf.AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[0, tlcf.SMM_I, :] = tlcf.cpr2smm(0.12)
    scenarios_tensor[0, tlcf.DQ_I, :] = 0
    
    mdrV = tlcf.cpr2smm(torch.tensor([0, 0, 0,])) # input refund_smm values
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
    dqV = .01 * torch.tensor([1, 2, 3])
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
    price_input = torch.tensor([[0.9381589717, 0.9020295044, 0.92732012833]])

    assert torch.allclose(price_result, price_input, rtol=0, atol=1e-8), "Advanced Price calculation mismatch"
    print("Non-advanced Price:", price_result)

    config_not_advance = tlcf.Config(is_advance=False)
    price_result = tlcf.y2p_tensor(loans_tensor, scenarios_tensor, yield_input, config_not_advance)
    price_input = torch.tensor([[0.93577210138, 0.87181853008, 0.90545638658]])

    assert torch.allclose(price_result, price_input, rtol=0, atol=1e-8), "Non-advanced Price calculation mismatch"
    print("Advanced Price:", price_result)


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    test_tensor_dqadvance()
