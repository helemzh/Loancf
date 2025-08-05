# Copyright (c) 2025 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy_financial as npf
import numpy as np
import torch
from collections import OrderedDict
from scipy.optimize import newton
from dataclasses import dataclass

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=12, linewidth=200)

cpr2smm = lambda cpr: 1-(1-cpr)**(1/12)
# Bond equivalent yield
bey2y = lambda y: 12 * ((1 + y / 2) ** (1 / 6) - 1)
y2bey = lambda y: 2 *((1 + y/12) ** 6 - 1)

def pad_rows(tensor, target_rows, pad_mode='zero'):
    """
    Pads the rows of a 2D tensor to target_rows by adding padding values either
    as zeros or repeating the last value of each column.

    Args:
        tensor (torch.Tensor): Input tensor of shape (M, W)
        target_rows (int): Target number of rows after padding
        pad_mode (str): 'zero' to pad with zeros (default), 'last' to pad with last value per column

    Returns:
        torch.Tensor: Padded tensor of shape (target_rows, W)
    """
    M, W = tensor.shape
    R = target_rows - M
    if R <= 0:
        return tensor[:target_rows, :]  # truncate if needed

    if pad_mode == 'zero':
        pad = torch.zeros(R, W, dtype=tensor.dtype, device=tensor.device)
    elif pad_mode == 'last':
        last_vals = tensor[-1, :].unsqueeze(0)
        pad = last_vals.repeat(R, 1)
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}. Use 'zero' or 'last'.")

    return torch.cat([tensor, pad], dim=0)

def pad_recovery_lag(v: torch.Tensor, recovery_lag: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Pads along dimension 1 (WL) by different left padding amounts given in recovery_lag per scenario (M) for each loan (N).

    Args:
        v (torch.Tensor): Input tensor of shape (N, WL, M)
        recovery_lag (torch.Tensor): Tensor of shape (1, M) or (M,), with non-negative integers for left-pad size per scenario
        max_len (int, optional): Output length along dim=1. If None, defaults to WL

    Returns:
        torch.Tensor: Output tensor of shape (N, WL, M), with zeros left-padded per scenario according to recovery_lag.
    """
    N, WL, M = v.shape
    # Ensure recovery_lag shape is (M,)
    recovery_lag = recovery_lag.view(-1).type(torch.int64)  # shape (M,)
    assert recovery_lag.shape[0] == M, "recovery_lag should have the same number of scenarios as v's last dim"

    if max_len is None:
        max_len = WL

    out = torch.zeros((N, max_len, M), dtype=v.dtype, device=v.device)

    # For each scenario m, pad all loans n the same way
    for m in range(M):
        lag = recovery_lag[m].item()
        valid_len = WL - max(lag, 0)
        if valid_len > 0:
            out[:, lag:lag+valid_len, m] = v[:, :valid_len, m]

    return out


@dataclass
class Config:
    mode: str = "exhaustive" # "exhaustive" or "matched"
    servicing_fee_method: str = "avg"  # "avg" or "beg"
    is_advance: bool = False
    agg_cf: bool = False


@dataclass
class ResultTensor:
    months: torch.Tensor
    totalPrinV: torch.Tensor
    schedPrinV: torch.Tensor
    prepayPrinV: torch.Tensor
    refundPrinV: torch.Tensor
    defaultV: torch.Tensor
    writeDownV: torch.Tensor
    recoveryV: torch.Tensor
    actInterestV: torch.Tensor
    servicingFeeV: torch.Tensor
    b_balanceV: torch.Tensor
    actualBalanceV: torch.Tensor
    cfV: torch.Tensor
    netCfV: torch.Tensor
    totalDefaultV: torch.Tensor


class LoanAmort(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def agg_tensor(result_tensor):
        """
        Aggregates all members of a ResultTensor object except 'months' by summing across dim=0 (loans) while keep dim
        """
        agg_fields = {}
        for field in result_tensor.__dataclass_fields__:
            val = getattr(result_tensor, field)
            if field == "months":
                agg_fields[field] = val
            else:
                agg_fields[field] = val.sum(dim=0, keepdim=True)
        return ResultTensor(**agg_fields)

    def forward(
            self, config, orig_wacV, wam, pv, rate_redV, refund_smmV, smmV, dqV, mdrV, sevV,
            aggMDR_timingV, aggMDR, compIntHC, servicing_fee, recovery_lag, 
            refund_premium, dq_adv_prin, dq_adv_int):
        """
        Takes in loan and scenario torch.tensors to calculate cashflow and result tensors

        Args:
            orig_wacV: [N, W]
            wam, pc: [N, 1]
            rate_redV, refund_smmV, smmV, dqV, mdrV, sevV, aggMDR_timingV: [W, M]
            aggMDR, compIntHC, servicing_fee, recovery_lag, refund_premium, dq_adv_prin, dq_adv_int: [1, M]

        Returns: ResultTensor Object with n_feature tensors of size [N, WL, M]
        """
        N, W = orig_wacV.shape
        M = smmV.shape[1]
        WL = int(torch.max(wam).item() + torch.max(recovery_lag).item())

        # pad tensors with shape W to WL (max wam + max recovery_lag) with zeros
        if WL != W:
            wacV = torch.cat([orig_wacV, torch.zeros(N, WL - W)], dim=1) # [N, WL]
            rate_redV = pad_rows(rate_redV, WL) # [WL, M]
            refund_smmV = pad_rows(refund_smmV, WL) # [WL, M]
            smmV = pad_rows(smmV, WL) # [WL, M]
            dqV = pad_rows(dqV, WL) # [WL, M]
            mdrV = pad_rows(mdrV, WL) # [WL, M]
            sevV = pad_rows(sevV, WL, pad_mode='last') # [WL, M]
            aggMDR_timingV = pad_rows(aggMDR_timingV, WL) # [WL, M]
        else:
            wacV = orig_wacV
        wacV = wacV.unsqueeze(2) - rate_redV.unsqueeze(0) # [N, WL, M]
        rateV = wacV / 12 # [N, WL, M]

        # Unfixed rate amortization
        exponent= torch.clamp(wam - torch.arange(WL).unsqueeze(0), min=1) # [N, WL], clamp 1 to avoid 0 in denominator for alpha
        alpha = rateV / ((1+rateV)**exponent.unsqueeze(2) - 1) 
        alpha = torch.cat([torch.zeros(N, 1, M), alpha], dim=1) # [N, WL+1, M], concat 0 to dimension 1
        balancesV = torch.nan_to_num(pv.repeat(1, WL+1).unsqueeze(2) * torch.cumprod(1-alpha, dim=1)) # [N, WL+1, M]
        balancesV[torch.abs(balancesV) <= 1e-12] = 0.0
        principalsV = balancesV[:, :-1, :] - balancesV[:, 1:, :] # [N, WL, M]
        interestsV = balancesV[:, :-1, :] * rateV # [N, WL, M]
        paydownV = torch.nan_to_num(principalsV / balancesV[:, :-1, :]) # [N, WL, M]

        p_survV = torch.cumprod(1 - smmV - refund_smmV - mdrV, dim=0) # [WL, M]
        default_aggMDRV = pv.view(N, 1, 1) * aggMDR.view(1, 1, M) * aggMDR_timingV.view(1, WL, M) # [N, WL, M]
        dqPrin_aggMDRV = paydownV * default_aggMDRV # [N, WL, M]
        scaled_default_aggMDRV = default_aggMDRV / (balancesV[:, :-1, :] * p_survV.unsqueeze(0) + 1e-16) # [N, WL, M]
        cum_scaled_default_aggMDRV = torch.cumsum(scaled_default_aggMDRV, dim=1) # [N, WL, M]
        survivorshipV = torch.cat([
            torch.ones(N, 1, M),
            p_survV.unsqueeze(0) * (1 - cum_scaled_default_aggMDRV)
        ], dim=1) # [N, WL+1, M]
        
        # Balances
        actualBalanceV = survivorshipV * balancesV # [N, WL+1, M]
        b_balanceV = actualBalanceV[:, :-1, :] # [N, WL, M], starts with month 0
        actualBalanceV = actualBalanceV[:, 1:, :] # [N, WL, M], starts with month 1

        # Scheduled Principals
        schedDQPrinV = survivorshipV[:, :-1, :] * principalsV * (1-mdrV.unsqueeze(0)) * dqV.unsqueeze(0) * (1-dq_adv_prin.unsqueeze(0)) # [N, WL, M]
        schedDefaultPrinV = survivorshipV[:, :-1, :] * principalsV * mdrV.unsqueeze(0) + dqPrin_aggMDRV # [N, WL, M]

        schedPrinV = survivorshipV[:, :-1, :] * principalsV - schedDQPrinV - schedDefaultPrinV # [N, WL, M]
        prepayPrinV = survivorshipV[:, :-1, :] * balancesV[:, 1:, :] * smmV.unsqueeze(0) # [N, WL, M]
        
        # Losses, recoveries, and writedowns
        defaultV = b_balanceV * mdrV.unsqueeze(0) + default_aggMDRV # [N, WL, M]
        writeDownV = pad_recovery_lag(defaultV, recovery_lag) # [N, WL, M]
        recoveryV = writeDownV * (1-sevV.unsqueeze(0)) # [N, WL, M]

        # Principals
        refundPrinV = survivorshipV[:, :-1, :] * balancesV[:, 1:, :] * refund_smmV.unsqueeze(0) # [N, WL, M]
        totalPrinV = schedPrinV + prepayPrinV + recoveryV # [N, WL, M]
        compIntV = prepayPrinV * rateV * compIntHC.unsqueeze(0) # [N, WL, M]
        refundIntV = refundPrinV * rateV # [N, WL, M]
        prepayPrinV = survivorshipV[:, :-1, :] * balancesV[:, 1:, :] * smmV.unsqueeze(0) + refundPrinV # [N, WL, M]

        # Servicing Fee
        defaultBalV = torch.clamp(torch.cumsum(defaultV - writeDownV, dim=1), min=0) # [N, WL, M]
        b_totalBalV = b_balanceV + torch.cat([torch.zeros(N, 1, M), defaultBalV[:, :-1, :]], dim=1) # [N, WL, M]
        totalBalV = actualBalanceV + defaultBalV # [N, WL, M]
        servicingFee_rate = servicing_fee.unsqueeze(0) / 12 # [M, 1]
        servicingFee_begV = b_totalBalV * servicingFee_rate # [N, WL, M]
        servicingFee_avgV = ((b_totalBalV + totalBalV) / 2) * servicingFee_rate # [N, WL, M]
        servicingFeeV = servicingFee_avgV if config.servicing_fee_method == 'avg' else servicingFee_begV

        actInterestV = rateV * b_balanceV if config.is_advance else (
            rateV * (b_balanceV * (1 - (1-mdrV.unsqueeze(0)) * dqV.unsqueeze(0) * (1-dq_adv_int.unsqueeze(0)) - mdrV.unsqueeze(0)) - default_aggMDRV) - compIntV)
        actInterestV -= refundIntV # [N, WL, M]

        cfV = totalPrinV + actInterestV # [N, WL, M]
        netCfV = torch.clamp(cfV - servicingFeeV, min=0) # [N, WL, M]
        totalDefaultV = schedDQPrinV + schedDefaultPrinV + defaultV # [N, WL, M]

        months = torch.arange(1, WL+1).unsqueeze(1).expand(N, WL, M) # [N, WL, M]

        if config.mode == "matched":
            # For each tensor of shape [N, WL, M], select [i, :, i] for i in range(N)
            def diag3d(tensor):
                idx = torch.arange(N)
                return tensor[idx, :, idx].unsqueeze(-1)  # [N, WL]
            months = diag3d(months)
            totalPrinV = diag3d(totalPrinV)
            schedPrinV = diag3d(schedPrinV)
            prepayPrinV = diag3d(prepayPrinV)
            refundPrinV = diag3d(refundPrinV)
            defaultV = diag3d(defaultV)
            writeDownV = diag3d(writeDownV)
            recoveryV = diag3d(recoveryV)
            actInterestV = diag3d(actInterestV)
            servicingFeeV = diag3d(servicingFeeV)
            b_balanceV = diag3d(b_balanceV)
            actualBalanceV = diag3d(actualBalanceV)
            cfV = diag3d(cfV)
            netCfV = diag3d(netCfV)
            totalDefaultV = diag3d(totalDefaultV)

        # Creating Result Object with tensors
        result_tensor = ResultTensor(months, totalPrinV, schedPrinV, prepayPrinV, refundPrinV, defaultV, writeDownV,
                                    recoveryV, actInterestV, servicingFeeV, b_balanceV, actualBalanceV, cfV, netCfV, totalDefaultV)

        if config.agg_cf:
            agg_tensor = LoanAmort.agg_tensor(result_tensor)
            return result_tensor, agg_tensor
        else:
            return result_tensor  
        

class Calc():
    def __init__(self, result_tensor, pv, config):
        self.result_tensor = result_tensor
        self.pv = pv
        self.config = config

    def y2p(self, y):
        N, WL, M = self.result_tensor.cfV.shape
        yV = (1 + y/12)**torch.arange(1, WL + 1) # [N, WL]
        if self.config.agg_cf and self.config.mode != "matched" and N == 1:
            pv = torch.sum(self.pv).expand(N, 1, M) # [1, 1, M]
        elif self.config.mode == "matched":
            pv = self.pv.reshape(N, 1, 1) # [N, 1, 1]
        else:
            pv = self.pv.unsqueeze(1).expand(N, 1, M) # [N, 1, M]
        px = torch.sum((self.result_tensor.cfV - self.result_tensor.servicingFeeV) / yV.unsqueeze(2), dim=1, keepdim=True) / (pv - torch.sum(self.result_tensor.refundPrinV / yV.unsqueeze(2), dim=1, keepdim=True)) # [N, 1, M]
        return px # [N, 1, M] price tensor
    
    def p2y(self, px, y_init=0.8):
        # change to numpy arrays
        cfV = self.result_tensor.cfV.cpu().numpy()
        servicingFeeV = self.result_tensor.servicingFeeV.cpu().numpy()
        refundPrinV = self.result_tensor.refundPrinV.cpu().numpy()
        pv = self.pv.cpu().numpy()

        n_loans = self.result_tensor.cfV.shape[0]
        m_scenarios = self.result_tensor.cfV.shape[2]
        months = np.arange(self.result_tensor.cfV.shape[1]) + 1

        yield_tensor = np.zeros((n_loans, m_scenarios))

        for i in range(n_loans):
            for j in range(m_scenarios):
                cf = cfV[i, :, j]
                fee = servicingFeeV[i, :, j]
                refund = refundPrinV[i, :, j]
                pv_ = float(np.sum(pv)) if m_scenarios == 1 else float(pv[j, 0].item())
                target_price = float(px[i, 0, j].cpu().numpy())

                def price_func(y):
                    yV = (1 + y / 12) ** months
                    numer = np.sum((cf - fee) / yV)
                    denom = pv_ - np.sum(refund / yV)
                    return numer / denom - target_price

                try:
                    y = newton(price_func, y_init) # initial guess yield
                except Exception:
                    y = np.nan
                yield_tensor[i, j] = y

        return torch.tensor(yield_tensor, dtype=torch.float64) # [N, M]

    def accr_int(wacV, rate_redV, settle_day):
        wacV = wacV.unsqueeze(2) - rate_redV.unsqueeze(0)
        return wacV * (settle_day - 1) / 360
