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
    prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV
    defaultV = b_balanceV * mdrV + default_aggMDRV

    # Losses, recoveries, and writedowns
    writedownV = defaultV
    recoveryV = defaultV - writedownV
    # shift for lag
    period_with_lag = periods + recovery_lag
    max_recovery_lag = int(recovery_lag.max().item())
    writedownV_pad = pad_zeros(writedownV, periods + max_recovery_lag)
    writedownV_shift = torch.zeros_like(writedownV_pad)
    # writedownV_pad: [N, T], recovery_lag: [N], T = periods + max_recovery_lag
    N, T = writedownV_pad.shape
    device = writedownV_pad.device
    lags = recovery_lag.long().clamp(min=0, max=T-1)  # ensure valid range

    # Create an index matrix for shifting
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, T)
    col_idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
    # For each row, subtract lag from col_idx to get source index
    src_idx = col_idx - lags.unsqueeze(1)
    # Mask for valid indices (src_idx >= 0)
    valid_mask = src_idx >= 0
    src_idx = src_idx.clamp(min=0)

    # Gather shifted values
    writedownV_shift[valid_mask] = writedownV_pad[row_idx[valid_mask], src_idx[valid_mask]]
    # Zero out positions before lag (already zero, but explicit)
    writedownV_shift[~valid_mask] = 0

    sevV_pad = pad_zeros(sevV, periods + recovery_lag, pad_value='last')
    recoveryV = writedownV_shift * (1 - sevV_pad)
    # Refund principal
    refundPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * refund_smm
    # Total principal
    schedPrinV_pad = pad_zeros(schedPrinV, periods + max_recovery_lag)
    prepayPrinV_pad = pad_zeros(prepayPrinV, periods + max_recovery_lag)
    totalPrinV = schedPrinV_pad + prepayPrinV_pad + recoveryV
    compIntV = prepayPrinV * rate.unsqueeze(1) * compIntHC.unsqueeze(-1)
    refundIntV = refundPrinV * rate.unsqueeze(1)
    prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV + refundPrinV

    # Servicing Fee
    defaultBalV = torch.cumsum(torch.cat([defaultV, torch.zeros(N, max_recovery_lag, device=device)], dim=1) - writedownV_shift, dim=1)
    b_totalBalV = torch.cat([b_balanceV, torch.zeros(N, max_recovery_lag, device=device)], dim=1) + torch.cat([defaultBalV[:, :-1], torch.zeros(N, 1, device=device)], dim=1)
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

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    # Example: 2 loans, 2 scenarios

    # loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
    loans_tensor = torch.tensor([
        [0.3, 12, 100],
        [0.04, 14, 150000]
    ], dtype=torch.float32)

    max_wam = int(loans_tensor[:, 1].max().item())
    n_scenarios = 2

    # [smm, dq, mdr, sev, refund_smm, aggMDR, aggMDR_timingV, compIntHC, servicing_fee, recovery_lag, refund_premium]
    
    ### Scenario 0 ###
    scenarios_tensor = torch.zeros(n_scenarios, 11, max_wam)
    scenarios_tensor[0, 0, :] = cpr2smm(0.35)  # scenario 1, cpr to smm conversion
    scenarios_tensor[0, 1, :] = 0     # dq
    scenarios_tensor[0, 2, :] = 0     # mdr
    scenarios_tensor[0, 3, :] = 0.94  # sev

    refund_smm = cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1])) # input refund_smm values
    scenarios_tensor[0, 4, :] = 0     # first fill 0
    scenarios_tensor[0, 4, :len(refund_smm)] = refund_smm 

    scenarios_tensor[0, 5, 0] = 0.03  # aggMDR (scalar, use first period)

    aggMDR_timingV = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1]) # input aggMDR_timingV values
    scenarios_tensor[0, 6, :] = 0     # aggMDR_timingV
    scenarios_tensor[0, 6, :len(aggMDR_timingV)] = aggMDR_timingV

    scenarios_tensor[0, 7, 0] = 0.2   # compIntHC (scalar)
    scenarios_tensor[0, 8, 0] = 0.02  # servicing_fee (scalar)
    scenarios_tensor[0, 9, 0] = 4     # recovery_lag (scalar)
    scenarios_tensor[0, 10, 0] = 1.0  # refund_premium (scalar)


    ### Scenario 1 ###
    scenarios_tensor[1, 0, :] = cpr2smm(0.35)  # cpr to smm conversion
    scenarios_tensor[1, 1, :] = 0     # dq
    scenarios_tensor[1, 2, :] = 0     # mdr
    scenarios_tensor[1, 3, :] = 0.94  # sev

    refund_smm = cpr2smm(.01 * torch.tensor([74, 15, 5, 3, 2, 1])) # input refund_smm values
    scenarios_tensor[1, 4, :] = 0     # first fill 0
    scenarios_tensor[1, 4, :len(refund_smm)] = refund_smm 

    scenarios_tensor[1, 5, 0] = 0.03  # aggMDR (scalar, use first period)

    aggMDR_timingV = .01 * torch.tensor([23,10,10,10,10,10,8,7,5,4,2,1]) # input aggMDR_timingV values
    scenarios_tensor[1, 6, :] = 0     # aggMDR_timingV
    scenarios_tensor[1, 6, :len(aggMDR_timingV)] = aggMDR_timingV

    scenarios_tensor[1, 7, 0] = 0.2   # compIntHC (scalar)
    scenarios_tensor[1, 8, 0] = 0.02  # servicing_fee (scalar)
    scenarios_tensor[1, 9, 0] = 4     # recovery_lag (scalar)
    scenarios_tensor[1, 10, 0] = 1.0  # refund_premium (scalar)

    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor)

    print("Result shape:", result_tensor.shape)  # [n_loans, n_scenarios, max_wam+lag, n_features]
    # print(f"Result for loan 0, scenario 0, all periods:\n\t{feature_names}\n{result_tensor[0, 0]}")
    print(f"Result for loan 0, scenario 1, all periods:\n\t{feature_names}\n{result_tensor[0, 1]}")