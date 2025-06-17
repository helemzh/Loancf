# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy_financial as npf
import numpy as np
import torch
from scipy.optimize import brentq


def cpr2smm(cpr):
    return 1 - (1 - cpr) ** (1 / 12)

def pad_zeros(vec, n, pad_value=0):
    """
    Pads each row of a 2D tensor to length n (scalar or [N]).
    If pad_value == 'last', pads with the last value of each row.
    Otherwise, pads with pad_value (default 0).
    vec: [N, T] or [T]
    n: int or [N] (target length per row)
    Returns: [N, max(n)] or [max(n)]
    """
    if isinstance(n, torch.Tensor):
        n = n.to(vec.device)
    if vec.ndim == 1:
        T = vec.shape[0]
        if isinstance(n, int):
            if T >= n:
                return vec[:n]
            pad_val = vec[-1] if pad_value == 'last' else pad_value
            pad = torch.full((n - T,), pad_val, dtype=vec.dtype, device=vec.device)
            return torch.cat([vec, pad])
        else:
            raise ValueError("For 1D vec, n must be an int")
    else:
        N, T = vec.shape
        if isinstance(n, int):
            if T >= n:
                return vec[:, :n]
            if pad_value == 'last':
                pad_vals = vec[:, -1].unsqueeze(1)  # [N, 1]
                pad = pad_vals.expand(N, n - T)
            else:
                pad = torch.full((N, n - T), pad_value, dtype=vec.dtype, device=vec.device)
            return torch.cat([vec, pad], dim=1)
        else:
            # n is a vector of length N
            n = n.long()
            max_n = int(n.max().item())
            out = torch.zeros((N, max_n), dtype=vec.dtype, device=vec.device)
            row_idx = torch.arange(N, device=vec.device).unsqueeze(1)
            col_idx = torch.arange(max_n, device=vec.device).unsqueeze(0)
            mask = col_idx < n.unsqueeze(1)
            gather_idx = torch.clamp(col_idx.expand(N, -1), max=T-1)
            if pad_value == 'last':
                out[mask] = vec.gather(1, gather_idx)[mask]
            else:
                out.fill_(pad_value)
                valid_cols = col_idx.expand(N, -1) < T
                out[mask & valid_cols] = vec.gather(1, gather_idx)[mask & valid_cols]
            return out

def shift_elements(arr, num, fill_value=0):
    # arr: [N, T]
    N, T = arr.shape
    result = torch.empty_like(arr)
    if num > 0:
        result[:, :num] = fill_value
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, num:] = fill_value
        result[:, :num] = arr[:, -num:]
    else:
        result[:] = arr
    return result

def safedivide(a, b):
    if np.isclose(b, 0, rtol=0, atol=3e-11):
        return 0
    else:
        return a / b

def calc(v): # for WAL calculations
    numerator = np.sum(np.maximum(0.0, v) * (np.arange(1, len(v) + 1) / 12.))
    denominator = np.sum(np.maximum(0.0, v))
    return safedivide(numerator, denominator) 

def pad_recovery_lag(v, recovery_lag, max_len=None):
    """
    v: [N, L] - input tensor (e.g., recovery or writedown)
    recovery_lag: [N] - number of zeros to pad at the start for each row
    max_len: int or None - total output length (default: max(recovery_lag + L))
    Returns: [N, max_len] tensor, each row left-padded by recovery_lag[i] zeros
    """
    N, L = v.shape
    if max_len is None:
        max_len = int((recovery_lag + L).max().item())
    out = torch.zeros(N, max_len, dtype=v.dtype, device=v.device)
    # Compute start indices for each row
    start_idx = recovery_lag
    # Compute indices for placing v into out
    row_idx = torch.arange(N, device=v.device).unsqueeze(1)         # [N, 1]
    col_idx = torch.arange(L, device=v.device).unsqueeze(0)         # [1, L]
    # Mask for valid placement (avoid overflow)
    valid = (start_idx.unsqueeze(1) + col_idx) < max_len            # [N, L]
    # Place values
    out[row_idx, start_idx.unsqueeze(1) + col_idx] = v * valid
    return out

def getCashflow_tensor(
    wac, wam, pv, smmV, dqV, mdrV, sevV, recovery_lag, refund_smm, refund_premium,
    aggMDR, aggMDR_timingV, compIntHC, servicing_fee, is_advance, servicing_fee_method
):
    """
    All arguments are torch tensors.
    wac, wam, pv: [N]
    smmV, dqV, mdrV, sevV, refund_smm, aggMDR_timingV: [N, periods]
    recovery_lag, refund_premium, aggMDR, compIntHC, servicing_fee: [N] or scalars
    is_advance: bool or [N]
    servicing_fee_method: str
    Returns: dict of tensors, each [N, periods+lag]
    """
    N, periods = smmV.shape
    # periods = wam if wam.ndim == 1 else wam.squeeze()
    wamV = wam.squeeze() if wam.ndim > 1 else wam  # [N]
    max_wam = int(wamV.max().item())
    device = wac.device

    rate = wac / 12
    X = -torch.tensor(
        npf.pmt(rate.cpu().numpy(), wamV.cpu().numpy(), pv.cpu().numpy()),
    )

    # Create period indices for all loans: [N, max_wamV+1]
    period_idx = torch.arange(max_wam + 1, device=device).unsqueeze(0).expand(N, -1)  # [N, max_wamV+1]
    mask = period_idx <= wamV.unsqueeze(1)  # [N, max_wamV+1]

    # Broadcast pv, rate, wam to [N, max_wamV+1]
    pv_exp = pv.unsqueeze(-1).expand(-1, max_wam + 1)
    rate_exp = rate.unsqueeze(-1).expand(-1, max_wam + 1)
    wam_exp = wamV.unsqueeze(-1).expand(-1, max_wam + 1)

    # Calculate balancesV for all loans and periods, then mask out padded positions
    balancesV = pv_exp * (1 - (1 + rate_exp) ** -(wam_exp - period_idx)) / (1 - (1 + rate_exp) ** -wam_exp)
    balancesV = balancesV * mask  # Zero out padded positions

    # Interests and principals
    interestsV = balancesV[:, :-1] * rate.unsqueeze(1)  # [N, periods-1]
    principalsV = X.unsqueeze(1) - interestsV  # [N, periods]
    paydownV = principalsV / balancesV[:, :-1]  # [N, periods]

    # Survivorship
    p_survV = torch.cumprod(1 - smmV - refund_smm - mdrV, dim=1)  # [N, periods]
    # Default aggregation
    default_aggMDRV = pv.view(-1, 1) * aggMDR.view(-1, 1) * aggMDR_timingV  # [N, periods]
    dqPrin_aggMDRV = paydownV * default_aggMDRV  # [N, periods]
    scaled_default_aggMDRV = default_aggMDRV / (balancesV[:, :-1] * p_survV + 1e-12)  # avoid div0
    cum_scaled_default_aggMDRV = torch.cumsum(scaled_default_aggMDRV, dim=1)
    survivorshipV = torch.cat([
        torch.ones(N, 1, device=device),
        p_survV * (1 - cum_scaled_default_aggMDRV)
    ], dim=1)  # [N, periods+1]

    actualBalanceV = survivorshipV * torch.cat([balancesV[:, :-1], balancesV[:, -1:].clone()], dim=1)  # [N, periods+1]
    b_balanceV = actualBalanceV[:, :-1]  # [N, periods]
    actualBalanceV = actualBalanceV[:, 1:]  # [N, periods]
    balanceDiffV = b_balanceV - actualBalanceV

    # Scheduled, prepayment, default, total principals, and deadbeat balance
    dqPrinV = torch.zeros_like(principalsV) if is_advance else (
        survivorshipV[:, :-1] * principalsV * (dqV + mdrV) + dqPrin_aggMDRV)
    schedPrinV = survivorshipV[:, :-1] * principalsV - dqPrinV
    schedPrinV = torch.nan_to_num(schedPrinV, nan=0.0)
    prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV
    defaultV = b_balanceV * mdrV + default_aggMDRV

    # Losses, recoveries, and writedowns
    writedownV = defaultV
    recoveryV = defaultV - writedownV
    period_with_lag = periods + recovery_lag
    max_recovery_lag = int(recovery_lag.max().item())
    writedownV_shift = pad_recovery_lag(writedownV, recovery_lag, max_len=periods + max_recovery_lag) # shift for lag
    sevV_pad = pad_zeros(sevV, periods + recovery_lag, pad_value='last')
    recoveryV = writedownV_shift * (1 - sevV_pad)
    
    # Principals
    refundPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * refund_smm
    schedPrinV_pad = pad_zeros(schedPrinV, periods + max_recovery_lag)
    prepayPrinV_pad = pad_zeros(prepayPrinV, periods + max_recovery_lag)
    totalPrinV = schedPrinV_pad + prepayPrinV_pad + recoveryV
    compIntV = prepayPrinV * rate.unsqueeze(1) * compIntHC.unsqueeze(-1)
    refundIntV = refundPrinV * rate.unsqueeze(1)
    prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV + refundPrinV

    # Servicing Fee Calculation
    defaultBalV = torch.cumsum(torch.cat([defaultV, torch.zeros(N, max_recovery_lag, device=device)], dim=1) - writedownV_shift, dim=1)
    defaultBalV = torch.clamp(defaultBalV, min=0)
    b_totalBalV = torch.cat([b_balanceV, torch.zeros(N, max_recovery_lag, device=device)], dim=1) + \
                  torch.cat([torch.zeros(N, 1, device=device), defaultBalV[:, :-1]], dim=1)
    totalBalV = torch.cat([actualBalanceV, torch.zeros(N, max_recovery_lag, device=device)], dim=1) + defaultBalV
    servicingFee_rate = servicing_fee.unsqueeze(-1) / 12
    servicingFee_begV = b_totalBalV * servicingFee_rate
    servicingFee_avgV = ((b_totalBalV + totalBalV) / 2) * servicingFee_rate
    servicingFeeV = servicingFee_avgV if servicing_fee_method == "avg" else servicingFee_begV

    # Interest and Cash Flow
    actInterestV = rate * b_balanceV if is_advance else (
        rate.unsqueeze(1) * (b_balanceV * (1 - (dqV + mdrV)) - default_aggMDRV) - compIntV)
    actInterestV -= refundIntV
    actInterestV_pad = torch.cat([actInterestV, torch.zeros(N, max_recovery_lag, device=device)], dim=1)
    cfV = totalPrinV + actInterestV_pad

    # Pad all vectors to period_with_lag
    def pad(v):
        if v.shape[1] < max_period_with_lag:
            return torch.cat([v, torch.zeros(N, max_period_with_lag - v.shape[1], device=device)], dim=1)
        return v
    
    max_period_with_lag = int(period_with_lag.max().item()) if torch.is_tensor(period_with_lag) else int(period_with_lag)

    # Output as dict of tensors
    return {
        "Months": torch.arange(1, max_period_with_lag + 1, device=device).unsqueeze(0).expand(N, -1),
        "Prin": pad(totalPrinV),
        "SchedPrin": pad(schedPrinV_pad),
        "Prepay Prin": pad(prepayPrinV_pad),
        "Refund Prin": pad(refundPrinV),
        "Default": pad(defaultV),
        "Writedown": pad(writedownV_shift),
        "Recovery": pad(recoveryV),
        "Interest": pad(actInterestV_pad),
        "Servicing Fee": pad(servicingFeeV),
        "Beginning Balance": pad(b_balanceV),
        "Balance": pad(actualBalanceV),
        "CFL": pad(cfV)
    }


class LoanAmort(torch.nn.Module):
    def __init__(self, loans_tensor):
        """
        loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
        """
        super().__init__()
        self.loans_tensor = loans_tensor
        self.n_loans = loans_tensor.shape[0]
        self.max_wam = int(self.loans_tensor[:, 1].max().item())

    def forward(self, scenarios_tensor):
        """
        scenarios_tensor: [n_scenarios, n_vectors, max_wam]
        n_vectors: [smm, dq, mdr, sev, refund_smm, aggMDR, aggMDR_timingV, compIntHC, servicing_fee]
        Returns: [n_loans, n_scenarios, max_wam+lag, n_features]
        """
        n_scenarios, n_vectors, max_wam = scenarios_tensor.shape
        n_loans = self.n_loans

        # Unpack loan attributes
        wac = self.loans_tensor[:, 0].view(n_loans, 1).expand(n_loans, n_scenarios).flatten()
        wam = self.loans_tensor[:, 1].long().view(n_loans, 1).expand(n_loans, n_scenarios).flatten()
        pv = self.loans_tensor[:, 2].view(n_loans, 1).expand(n_loans, n_scenarios).flatten()

        # Unpack scenario attributes
        smmV = scenarios_tensor[:, 0, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        dqV = scenarios_tensor[:, 1, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        mdrV = scenarios_tensor[:, 2, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        sevV = scenarios_tensor[:, 3, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        refund_smm = scenarios_tensor[:, 4, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        aggMDR = scenarios_tensor[:, 5, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
        aggMDR_timingV = scenarios_tensor[:, 6, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
        compIntHC = scenarios_tensor[:, 7, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
        servicing_fee = scenarios_tensor[:, 8, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
        recovery_lag = scenarios_tensor[:, 9, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten().int()
        refund_premium = scenarios_tensor[:, 10, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()

        # Set these as constants for all scenarios/loans for simplicity
        is_advance = False
        servicing_fee_method = "avg"

        # Call vectorized cashflow for all pairs
        cf_dict = getCashflow_tensor(
            wac, wam, pv, smmV, dqV, mdrV, sevV, recovery_lag, refund_smm, refund_premium,
            aggMDR, aggMDR_timingV, compIntHC, servicing_fee, is_advance, servicing_fee_method
        )

        # Stack results into a tensor: [n_pairs, period_with_lag, n_features]
        feature_names = list(cf_dict.keys())
        n_features = len(feature_names)
        period_with_lag = cf_dict["CFL"].shape[1]
        n_pairs = n_loans * n_scenarios
        result_tensor = torch.stack([cf_dict[k] for k in feature_names], dim=-1)  # [n_pairs, period_with_lag, n_features]
        result_tensor = result_tensor.view(n_loans, n_scenarios, period_with_lag, n_features)
        return result_tensor, feature_names

def y2p_tensor(loans_tensor, scenarios_tensor, yield_tensor):
    """
    Vectorized yield-to-price for all loans and scenarios, using each loan/scenario's actual wam+lag.
    loans_tensor: [n_loans, 3]
    scenarios_tensor: [n_scenarios, n_vectors, max_wam]
    yield_tensor: [n_loans, n_scenarios] (annualized yield)
    Returns: price_tensor [n_loans, n_scenarios]
    """
    n_loans, _ = loans_tensor.shape
    n_scenarios = scenarios_tensor.shape[0]
    max_wam = int(loans_tensor[:, 1].max().item())
    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor)  # [n_loans, n_scenarios, max_wam+lag, n_features]

    # Get actual wam and lag for each loan/scenario
    wam = loans_tensor[:, 1].long().unsqueeze(1).expand(-1, n_scenarios)  # [n_loans, n_scenarios]
    lag = scenarios_tensor[:, RECOVERY_LAG_I, 0].long().unsqueeze(0).expand(n_loans, -1)  # [n_loans, n_scenarios]
    actual_len = wam + lag  # [n_loans, n_scenarios]
    max_len = result_tensor.shape[2]

    # Build mask for valid periods
    period_idx = torch.arange(max_len, device=result_tensor.device).view(1, 1, -1)
    mask = period_idx < actual_len.unsqueeze(-1)  # [n_loans, n_scenarios, max_len]

    # Discount factors
    months = period_idx + 1  # [1, 1, max_len]
    yV = (1 + yield_tensor.unsqueeze(-1) / 12) ** months  # [n_loans, n_scenarios, max_len]
    cfV = result_tensor[..., feature_names.index("CFL")]           # [n_loans, n_scenarios, max_len]
    servicingFeeV = result_tensor[..., feature_names.index("Servicing Fee")]
    refundPrinV = result_tensor[..., feature_names.index("Refund Prin")]

    pv = loans_tensor[:, 2].unsqueeze(1)         # [n_loans, 1]

    # Discounted cash flows, only sum over valid periods
    numer = torch.sum(((cfV - servicingFeeV) / yV), dim=-1)  # [n_loans, n_scenarios]
    denom = pv - torch.sum((refundPrinV / yV) * mask, dim=-1)       # [n_loans, n_scenarios]
    px = (numer / denom)
    return px

def p2y_tensor(loans_tensor, scenarios_tensor, price_tensor, y_min=0.0001, y_max=1.0, tol=1e-6, max_iter=100):
    """
    Vectorized price-to-yield for all loans and scenarios using bisection.
    loans_tensor: [n_loans, 3]
    scenarios_tensor: [n_scenarios, n_vectors, max_wam]
    price_tensor: [n_loans, n_scenarios]
    Returns: yield_tensor [n_loans, n_scenarios]
    """
    n_loans, n_scenarios = price_tensor.shape
    device = loans_tensor.device

    # Initial bounds
    y_lo = torch.full((n_loans, n_scenarios), y_min, device=device)
    y_hi = torch.full((n_loans, n_scenarios), y_max, device=device)

    for _ in range(max_iter):
        y_mid = (y_lo + y_hi) / 2
        px_mid = y2p_tensor(loans_tensor, scenarios_tensor, y_mid)
        above = px_mid > price_tensor
        y_lo = torch.where(above, y_mid, y_lo)
        y_hi = torch.where(~above, y_mid, y_hi)
        if torch.max(torch.abs(px_mid - price_tensor)) < tol:
            break
    return y_mid


# Scenario tensor index constants
SMM_I = 0
DQ_I = 1
MDR_I = 2
SEV_I = 3
REFUND_SMM_I = 4
AGGMDR_I = 5
AGGMDR_TIMING_I = 6
COMPINTHC_I = 7
SERVICING_FEE_I = 8
RECOVERY_LAG_I = 9
REFUND_PREMIUM_I = 10

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    # Example: 2 loans, 2 scenarios

    # loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
    loans_tensor = torch.tensor([
        [0.3, 12, 100],
        [0.3, 14, 100],
    ], dtype=torch.float32)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_loans = loans_tensor.shape[0]
    n_scenarios = 2
    
    scenarios_tensor = torch.zeros(n_scenarios, 11, max_wam)

    ### Scenario 0 ###
    # Set up refund_smm and aggMDR_timingV
    refund_smm = cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1])) # input refund_smm values
    scenarios_tensor[0, REFUND_SMM_I, :] = 0 # first fill 0
    scenarios_tensor[0, REFUND_SMM_I, :len(refund_smm)] = refund_smm
    aggMDR_timing = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1]) # input aggMDR_timing values
    scenarios_tensor[0, AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[0, AGGMDR_TIMING_I, :len(aggMDR_timing)] = aggMDR_timing
    scenarios_tensor[0, SMM_I, :] = cpr2smm(0.35)
    scenarios_tensor[0, DQ_I, :] = 0
    scenarios_tensor[0, MDR_I, :] = 0
    scenarios_tensor[0, SEV_I, :] = 0.94
    scenarios_tensor[0, AGGMDR_I, 0] = 0.03
    scenarios_tensor[0, COMPINTHC_I, 0] = 0.2
    scenarios_tensor[0, SERVICING_FEE_I, 0] = 0.02
    scenarios_tensor[0, RECOVERY_LAG_I, 0] = 4
    scenarios_tensor[0, REFUND_PREMIUM_I, 0] = 1.0

    ### Scenario 1 ###
    refund_smm = cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1])) # input refund_smm values
    scenarios_tensor[1, REFUND_SMM_I, :] = 0 # first fill 0
    scenarios_tensor[1, REFUND_SMM_I, :len(refund_smm)] = refund_smm
    aggMDR_timing = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1]) # input aggMDR_timing values
    scenarios_tensor[1, AGGMDR_TIMING_I, :] = 0
    scenarios_tensor[1, AGGMDR_TIMING_I, :len(aggMDR_timing)] = aggMDR_timing
    scenarios_tensor[1, SMM_I, :] = cpr2smm(0.35)
    scenarios_tensor[1, DQ_I, :] = 0
    scenarios_tensor[1, MDR_I, :] = 0
    scenarios_tensor[1, SEV_I, :] = 0.94
    scenarios_tensor[1, AGGMDR_I, 0] = 0.03
    scenarios_tensor[1, COMPINTHC_I, 0] = 0.2
    scenarios_tensor[1, SERVICING_FEE_I, 0] = 0.02
    scenarios_tensor[1, RECOVERY_LAG_I, 0] = 3
    scenarios_tensor[1, REFUND_PREMIUM_I, 0] = 1.0

    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor)

    print("Result shape:", result_tensor.shape)  # [n_loans, n_scenarios, max_wam+lag, n_features]
    print(f"Result for loan 0, scenario 0:\n\t{feature_names}\n{result_tensor[0, 0]}")
    print(f"Result for loan 0, scenario 1:\n\t{feature_names}\n{result_tensor[0, 1]}")
    print(f"Result for loan 1, scenario 0:\n\t{feature_names}\n{result_tensor[1, 0]}")
    print(f"Result for loan 1, scenario 1:\n\t{feature_names}\n{result_tensor[1, 1]}")

    # Suppose you want price for a given yield:
    yield_tensor = torch.full((n_loans, n_scenarios), 0.1, device=loans_tensor.device)
    price_tensor = y2p_tensor(loans_tensor, scenarios_tensor, yield_tensor)

    # Or, want yield for a given price:
    target_price = torch.full((n_loans, n_scenarios), 1.061202911, device=loans_tensor.device) #1.0506307058881
    yield_tensor = p2y_tensor(loans_tensor, scenarios_tensor, target_price)

    print(f"price{price_tensor}, \nyield{yield_tensor}")