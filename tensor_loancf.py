# Copyright (c) 2025 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy_financial as npf
import numpy as np
import torch
from scipy.optimize import newton
from dataclasses import dataclass

# Loan tensor index constants
L_WAC_I = 0
L_WAM_I = 1
L_PV_I = 2

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
RATE_RED_I = 11
DQ_ADV_PRIN_I = 12
DQ_ADV_INT_I = 13

# Result tensor index constants
RESULT_MONTHS_I = 0
RESULT_PRIN_I = 1
RESULT_SCHED_PRIN_I = 2
RESULT_PREPAY_PRIN_I = 3
RESULT_REFUND_PRIN_I = 4
RESULT_DEFAULT_I = 5
RESULT_WRITEDOWN_I = 6
RESULT_RECOVERY_I = 7
RESULT_INTEREST_I = 8
RESULT_SERVICING_FEE_I = 9
RESULT_BEGINNING_BALANCE_I = 10
RESULT_BALANCE_I = 11
RESULT_CFL_I = 12
RESULT_TOTAL_DEFAULT_I = 13

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

cpr2smm = lambda cpr: 1-(1-cpr)**(1/12)
bey2y = lambda y: 12 * ((1 + y / 2) ** (1 / 6) - 1)
y2bey = lambda y: 2 *((1 + y/12) ** 6 - 1)

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


@dataclass
class Config:
    mode: str = "exhaustive"                   # "exhaustive" or "matched" 
    servicing_fee_method: str = "avg"          # "avg" or "beg"
    rate_red_method: bool = False              # default False = unfixed rate
    is_advance: bool = False                  # False unless otherwise


def getCashflow_tensor(
    config, wac, wam, pv, smmV, dqV, mdrV, sevV, recovery_lag, refund_smm, refund_premium,
    aggMDR, aggMDR_timingV, compIntHC, servicing_fee, rate_redV, dq_adv_prin, dq_adv_int
):
    """
    All arguments are torch tensors.
    Args: 
        wac, wam, pv: [N]
        rate_redV, smmV, dqV, mdrV, sevV, refund_smm, aggMDR_timingV: [N, periods]
        recovery_lag, refund_premium, aggMDR, compIntHC, servicing_fee: [N] or scalars
    Returns: 
        dict of tensors, each [N, periods+lag]
    """
    N, periods = smmV.shape

    wamV = wam.squeeze() if wam.ndim > 1 else wam  # [N]
    pv = pv.squeeze()
    wac = wac.squeeze()
    max_wam = int(wamV.max().item())

    wacV = wac.unsqueeze(1) - rate_redV
    rateV = wacV / 12

    # Create period indices for all loans: [N, max_wamV+1]
    period_idx = torch.arange(max_wam + 1, device=device).unsqueeze(0).expand(N, -1)  # [N, max_wamV+1]
    mask = period_idx <= wamV.unsqueeze(1)  # [N, max_wamV+1]

    # Broadcast pv, rate, wam to [N, max_wamV+1]
    pv_exp = pv.unsqueeze(-1).expand(-1, max_wam + 1)
    rate_exp = torch.cat((rateV, rateV[..., -1:].clone()), dim=-1)
    wam_exp = wamV.unsqueeze(-1).expand(-1, max_wam + 1)

    # Calculate balancesV for all loans and periods, then mask out padded positions
    if config.rate_red_method == False:
        # Fixed rate calculation
        X = (pv_exp * rate_exp) / (1 - (1 + rate_exp) ** -wam_exp)
        balancesV = pv_exp * (1 - (1 + rate_exp) ** -(wam_exp - period_idx)) / (1 - (1 + rate_exp) ** -wam_exp)
        balancesV = balancesV * mask  # Zero out padded positions

        interestsV = balancesV[:, :-1] * rateV  # [N, periods-1]
        principalsV = X[:, :-1] - interestsV  # [N, periods]
        paydownV = principalsV / balancesV[:, :-1]  # [N, periods]
    else:
        remaining_terms = wam_exp - period_idx
        denominator = (1 + rate_exp) ** remaining_terms - 1
        alphaV = rate_exp / denominator
        alphaV = torch.nan_to_num(alphaV, nan=0.0, posinf=0.0, neginf=0.0)
        cumprod = torch.cumprod(1 - alphaV, dim=1)
        balancesV = torch.cat([pv_exp[:, :1], pv_exp[:, :-1] * cumprod[:, :-1]], dim=1)
        balancesV = torch.maximum(balancesV, torch.zeros_like(balancesV))  # Ensure non-negative balances
        principalsV = balancesV[:, :-1] - balancesV[:, 1:]
        interestsV = balancesV[:, :-1] * rate_exp[:, :-1]
        paydownV = principalsV / balancesV[:, :-1]

    # Survivorship
    p_survV = torch.cumprod(1 - smmV - refund_smm - mdrV, dim=1)  # [N, periods]
    # Default aggregation
    default_aggMDRV = pv.view(-1, 1) * aggMDR.view(-1, 1) * aggMDR_timingV  # [N, periods]
    dqPrin_aggMDRV = paydownV * default_aggMDRV  # [N, periods]
    scaled_default_aggMDRV = default_aggMDRV / (balancesV[:, :-1] * p_survV + 1e-16)  # avoid div0
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
    ''' old calculation
    dqPrinV = torch.zeros_like(principalsV) if is_advance else (
        survivorshipV[:, :-1] * principalsV * (dqV + mdrV) + dqPrin_aggMDRV)
    '''
    schedDQPrinV = survivorshipV[:, :-1] * principalsV * (1-mdrV) * dqV * (1-dq_adv_prin)
    schedDefaultPrinV = survivorshipV[:, :-1] * principalsV * mdrV + dqPrin_aggMDRV
    
    schedPrinV = survivorshipV[:, :-1] * principalsV - schedDQPrinV - schedDefaultPrinV
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
    totalPrinV = schedPrinV_pad + prepayPrinV_pad
    compIntV = prepayPrinV * rateV * compIntHC.unsqueeze(-1)
    refundIntV = refundPrinV * rateV
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
    servicingFeeV = servicingFee_avgV if config.servicing_fee_method == "avg" else servicingFee_begV

    # Interest and Cash Flow
    actInterestV = rateV * b_balanceV if config.is_advance else (
        rateV * (b_balanceV * (1- (1-mdrV) * dqV*(1-dq_adv_int) - mdrV) - default_aggMDRV) - compIntV)
    actInterestV -= refundIntV

    if max_recovery_lag > 0:
        actInterestV_pad = torch.cat(
            [actInterestV, torch.zeros(actInterestV.shape[0], max_recovery_lag, device=device)], dim=1
        )
    else:
        actInterestV_pad = actInterestV

    cfV = totalPrinV + actInterestV_pad
    totalDefaultV = schedDQPrinV + schedDefaultPrinV + defaultV

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
        "CFL": pad(cfV),
        "Total Default": pad(totalDefaultV),
    }


class LoanAmort(torch.nn.Module):
    def __init__(self, loans_tensor):
        """
        Args:
            loans_tensor: [n_loans, 3] (columns: wac, wam, pv)
        """
        super().__init__()
        self.loans_tensor = loans_tensor
        self.n_loans = loans_tensor.shape[0]
        self.max_wam = int(self.loans_tensor[:, 1].max().item())

    def forward(self, scenarios_tensor, config):
        """
        Args:
            scenarios_tensor: [n_scenarios, n_attributes, max_wam]
            mode: "exhaustive" or "matched"
            servicing_fee_method: "avg" or "beg"
            rate_red_method: True or False (unfixed or fixed rate)
            is_advance: bool
            n_vectors: [smm, dq, mdr, sev, refund_smm, aggMDR, aggMDR_timingV, compIntHC, servicing_fee]
        Returns: 
            [n_loans, n_scenarios, max_wam+lag, n_features]
        """
        n_scenarios, n_vectors, max_wam = scenarios_tensor.shape
        n_loans = self.n_loans

        if config.mode == "matched":
            assert n_loans == n_scenarios, "Matched mode requires n_loans == n_scenarios"
            # Loan attributes
            wac = self.loans_tensor[:, L_WAC_I]
            wam = self.loans_tensor[:, L_WAM_I].long()
            pv = self.loans_tensor[:, L_PV_I]

            # Scenario attributes
            smmV = scenarios_tensor[:, SMM_I, :]
            dqV = scenarios_tensor[:, DQ_I, :]
            mdrV = scenarios_tensor[:, MDR_I, :]
            sevV = scenarios_tensor[:, SEV_I, :]
            refund_smm = scenarios_tensor[:, REFUND_SMM_I, :]
            aggMDR = scenarios_tensor[:, AGGMDR_I, 0]
            aggMDR_timingV = scenarios_tensor[:, AGGMDR_TIMING_I, :]
            compIntHC = scenarios_tensor[:, COMPINTHC_I, 0]
            servicing_fee = scenarios_tensor[:, SERVICING_FEE_I, 0]
            recovery_lag = scenarios_tensor[:, RECOVERY_LAG_I, 0].int()
            refund_premium = scenarios_tensor[:, REFUND_PREMIUM_I, 0]
            rate_redV = scenarios_tensor[:, RATE_RED_I, :]
            dq_adv_prin = scenarios_tensor[:, DQ_ADV_PRIN_I, :]
            dq_adv_int = scenarios_tensor[:, DQ_ADV_INT_I, :]

        else:
            # Unpack loan attributes
            wac = self.loans_tensor[:, L_WAC_I].view(n_loans, 1).expand(n_loans, n_scenarios).flatten()
            wam = self.loans_tensor[:, L_WAM_I].long().view(n_loans, 1).expand(n_loans, n_scenarios).flatten()
            pv = self.loans_tensor[:, L_PV_I].view(n_loans, 1).expand(n_loans, n_scenarios).flatten()

            # Unpack scenario attributes
            smmV = scenarios_tensor[:, SMM_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            dqV = scenarios_tensor[:, DQ_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            mdrV = scenarios_tensor[:, MDR_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            sevV = scenarios_tensor[:, SEV_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            refund_smm = scenarios_tensor[:, REFUND_SMM_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            aggMDR = scenarios_tensor[:, AGGMDR_I, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
            aggMDR_timingV = scenarios_tensor[:, AGGMDR_TIMING_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            compIntHC = scenarios_tensor[:, COMPINTHC_I, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
            servicing_fee = scenarios_tensor[:, SERVICING_FEE_I, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
            recovery_lag = scenarios_tensor[:, RECOVERY_LAG_I, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten().int()
            refund_premium = scenarios_tensor[:, REFUND_PREMIUM_I, 0].unsqueeze(0).expand(n_loans, n_scenarios).flatten()
            rate_redV = scenarios_tensor[:, RATE_RED_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            dq_adv_prin = scenarios_tensor[:, DQ_ADV_PRIN_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)
            dq_adv_int = scenarios_tensor[:, DQ_ADV_INT_I, :].unsqueeze(0).expand(n_loans, n_scenarios, max_wam).reshape(-1, max_wam)

        # Call vectorized cashflow for all pairs
        cf_dict = getCashflow_tensor(
            config, wac, wam, pv, smmV, dqV, mdrV, sevV, recovery_lag, refund_smm, refund_premium,
            aggMDR, aggMDR_timingV, compIntHC, servicing_fee, rate_redV, dq_adv_prin, dq_adv_int
        )

        # Stack results into a tensor: [n_pairs, period_with_lag, n_features]
        feature_names = list(cf_dict.keys())
        n_features = len(feature_names)
        period_with_lag = cf_dict["CFL"].shape[1]
        if config.mode == "matched":
            result_tensor = torch.stack([cf_dict[k] for k in feature_names], dim=-1)  # [n_scenarios, period_with_lag, n_features]
            return result_tensor, feature_names
        else:
            n_pairs = n_loans * n_scenarios
            result_tensor = torch.stack([cf_dict[k] for k in feature_names], dim=-1)  # [n_pairs, period_with_lag, n_features]
            result_tensor = result_tensor.view(n_loans, n_scenarios, period_with_lag, n_features)
            return result_tensor, feature_names

def y2p_tensor(loans_tensor, scenarios_tensor, yield_tensor, config):
    """
    Vectorized yield-to-price for all loans and scenarios, using each loan/scenario's actual wam+lag.
    Args:
        loans_tensor: [n_loans, 3]
        scenarios_tensor: [n_scenarios, n_vectors, max_wam]
        yield_tensor: [n_loans, n_scenarios] (annualized yield)
        config: Config object
    Returns: 
        price_tensor: [n_loans, n_scenarios]
    """
    n_loans, _ = loans_tensor.shape
    n_scenarios = scenarios_tensor.shape[0]
    max_wam = int(loans_tensor[:, 1].max().item())
    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor, config)  # [n_loans, n_scenarios, max_wam+lag, n_features]

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
    recoveryV = result_tensor[..., feature_names.index("Recovery")]

    pv = loans_tensor[:, 2].unsqueeze(1)         # [n_loans, 1]

    # Discounted cash flows, only sum over valid periods
    numer = torch.sum(((cfV + recoveryV - servicingFeeV) / yV), dim=-1)  # [n_loans, n_scenarios]
    denom = pv - torch.sum((refundPrinV / yV) * mask, dim=-1)       # [n_loans, n_scenarios]
    px = (numer / denom)
    return px

def p2y_tensor(loans_tensor, scenarios_tensor, price_tensor, config, y_init=.08):
    """
    Args:
        loans_tensor: [n_loans, 3] (wac, wam, pv)
        scenarios_tensor: [n_scenarios, n_vectors, max_wam]
        price_tensor: [n_loans, n_scenarios]
        config: Config object
    Returns: 
        yield_tensor [n_loans, n_scenarios]
    """
    n_loans, n_scenarios = price_tensor.shape
    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor, config)  # [n_loans, n_scenarios, periods, n_features]
    periods = result_tensor.shape[2]
    months = np.arange(1, periods + 1)

    cfV = result_tensor[..., feature_names.index("CFL")].cpu().numpy()
    servicingFeeV = result_tensor[..., feature_names.index("Servicing Fee")].cpu().numpy()
    refundPrinV = result_tensor[..., feature_names.index("Refund Prin")].cpu().numpy()
    recoveryV = result_tensor[..., feature_names.index("Recovery")].cpu().numpy()
    pv = loans_tensor[:, 2].cpu().numpy()

    yield_tensor = np.zeros((n_loans, n_scenarios))

    for i in range(n_loans):
        for j in range(n_scenarios):
            cf = cfV[i, j]
            fee = servicingFeeV[i, j]
            refund = refundPrinV[i, j]
            recovery = recoveryV[i, j]
            pv_ = float(pv[i])
            target_price = float(price_tensor[i, j].cpu().numpy())

            def price_func(y):
                yV = (1 + y / 12) ** months
                numer = np.sum((cf + recovery - fee) / yV)
                denom = pv_ - np.sum(refund / yV)
                return numer / denom - target_price

            try:
                y = newton(price_func, y_init) # initial guess yield
            except Exception:
                y = np.nan
            yield_tensor[i, j] = y

    return torch.tensor(yield_tensor, dtype=loans_tensor.dtype, device=loans_tensor.device)


# Outdated, will be adjusted for the new matched/exhaustive mode
def yield_price_aggregation(
    loans_tensor,
    scenarios_tensor,
    value_tensor,        # yield_tensor for y2p, price_tensor for p2y
    function='y2p',      # 'y2p' or 'p2y'
    method='pool',       # 'pool' or 'loan'
    y_init=0.05
):
    """
    Args:
        loans_tensor: [n_loans, 3]
        scenarios_tensor: [n_scenarios, n_vectors, max_wam]
        value_tensor: [n_loans, n_scenarios] or [n_scenarios] or [n_loans] (yield or price)
        function: 'y2p' or 'p2y'
        method: 'pool' or 'loan'
        y_init: initial guess for yield (for p2y)
    Returns: 
        [n_scenarios] for pool, [n_loans] for loan (1-to-1)
    """
    model = LoanAmort(loans_tensor)
    result_tensor, feature_names = model(scenarios_tensor)  # [n_loans, n_scenarios, periods, n_features]
    cfV = result_tensor[..., feature_names.index("CFL")]
    servicingFeeV = result_tensor[..., feature_names.index("Servicing Fee")]
    refundPrinV = result_tensor[..., feature_names.index("Refund Prin")]
    pv = loans_tensor[:, 2]

    n_loans, n_scenarios, periods = cfV.shape

    if method == 'loan':
        # 1-to-1 mapping: n_loans == n_scenarios
        if n_loans != n_scenarios:
            raise ValueError("For loan method, n_loans must equal n_scenarios for 1-to-1 mapping.")
        idx = torch.arange(n_loans, device=cfV.device)
        cfV_diag = cfV[idx, idx, :]                # [n_loans, periods]
        servicingFeeV_diag = servicingFeeV[idx, idx, :]
        refundPrinV_diag = refundPrinV[idx, idx, :]
        pv_diag = pv[idx]
        periods = cfV_diag.shape[1]
        months = torch.arange(1, periods + 1, device=cfV_diag.device).unsqueeze(0)  # [1, periods]

        if function == 'y2p':
            yield_vec = value_tensor.flatten()
            if yield_vec.shape[0] != n_loans:
                raise ValueError("For loan y2p, value_tensor must have length equal to number of loans.")
            yV = (1 + yield_vec.unsqueeze(1) / 12) ** months  # [n_loans, periods]
            numer = torch.sum((cfV_diag - servicingFeeV_diag) / yV, dim=1)  # [n_loans]
            denom = pv_diag - torch.sum(refundPrinV_diag / yV, dim=1)       # [n_loans]
            price = numer / denom
            return price

        elif function == 'p2y':
            price_vec = value_tensor.flatten()
            if price_vec.shape[0] != n_loans:
                raise ValueError("For loan p2y, value_tensor must have length equal to number of loans.")
            yield_vec = np.zeros(n_loans)
            months_np = np.arange(1, periods + 1)
            cfV_np = cfV_diag.cpu().numpy()
            fee_np = servicingFeeV_diag.cpu().numpy()
            refund_np = refundPrinV_diag.cpu().numpy()
            pv_np = pv_diag.cpu().numpy()
            price_np = price_vec.cpu().numpy()
            for i in range(n_loans):
                cf = np.nan_to_num(cfV_np[i], nan=0.0)
                fee = np.nan_to_num(fee_np[i], nan=0.0)
                refund = np.nan_to_num(refund_np[i], nan=0.0)
                pv_ = float(pv_np[i])
                target_price = float(price_np[i])
                def price_func(y):
                    yV = (1 + y / 12) ** months_np
                    numer = np.sum((cf - fee) / yV)
                    denom = pv_ - np.sum(refund / yV)
                    return numer / denom - target_price
                try:
                    y = newton(price_func, y_init)
                except Exception:
                    y = np.nan
                yield_vec[i] = y
            return torch.tensor(yield_vec, dtype=loans_tensor.dtype, device=loans_tensor.device)
        else:
            raise ValueError("function must be 'y2p' or 'p2y'")

    elif method == 'pool':
        # Aggregate all loans per scenario
        agg_cfV = cfV.sum(dim=0)                # [n_scenarios, periods]
        agg_servicingFeeV = servicingFeeV.sum(dim=0)
        agg_refundPrinV = refundPrinV.sum(dim=0)
        agg_pv = pv.sum().expand(n_scenarios)   # [n_scenarios]
        periods = agg_cfV.shape[1]
        months = torch.arange(1, periods + 1, device=agg_cfV.device).unsqueeze(0)  # [1, periods]

        if function == 'y2p':
            yield_vec = value_tensor.flatten()
            if yield_vec.shape[0] == 1:
                yield_vec = yield_vec.expand(n_scenarios)
            if yield_vec.shape[0] != n_scenarios:
                raise ValueError("For pool y2p, value_tensor must have length equal to number of scenarios or 1.")
            yV = (1 + yield_vec.unsqueeze(1) / 12) ** months  # [n_scenarios, periods]
            numer = torch.sum((agg_cfV - agg_servicingFeeV) / yV, dim=1)  # [n_scenarios]
            denom = agg_pv - torch.sum(agg_refundPrinV / yV, dim=1)       # [n_scenarios]
            price = numer / denom
            return price

        elif function == 'p2y':
            price_vec = value_tensor.flatten()
            if price_vec.shape[0] == 1:
                price_vec = price_vec.expand(n_scenarios)
            if price_vec.shape[0] != n_scenarios:
                raise ValueError("For pool p2y, value_tensor must have length equal to number of scenarios or 1.")
            yield_vec = np.zeros(n_scenarios)
            months_np = np.arange(1, periods + 1)
            for i in range(n_scenarios):
                cf = np.nan_to_num(agg_cfV[i].cpu().numpy(), nan=0.0)
                fee = np.nan_to_num(agg_servicingFeeV[i].cpu().numpy(), nan=0.0)
                refund = np.nan_to_num(agg_refundPrinV[i].cpu().numpy(), nan=0.0)
                pv_ = float(agg_pv[i].cpu().numpy())
                target_price = float(price_vec[i])
                def price_func(y):
                    yV = (1 + y / 12) ** months_np
                    numer = np.sum((cf - fee) / yV)
                    denom = pv_ - np.sum(refund / yV)
                    return numer / denom - target_price
                try:
                    y = newton(price_func, y_init)
                except Exception:
                    y = np.nan
                yield_vec[i] = y
            return torch.tensor(yield_vec, dtype=loans_tensor.dtype, device=loans_tensor.device)
        else:
            raise ValueError("function must be 'y2p' or 'p2y'")

    else:
        raise ValueError("method must be 'pool' or 'loan'")
        
