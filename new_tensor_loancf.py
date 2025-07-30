# Copyright (c) 2024 Helen Zhang <zhangdhelen@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy_financial as npf
import numpy as np
import torch
from scipy.optimize import newton
from dataclasses import dataclass

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=12, linewidth=200)

cpr2smm = lambda cpr: 1-(1-cpr)**(1/12)

# Bond equivalent yield
bey2y = lambda y: 12 * ((1 + y / 2) ** (1 / 6) - 1)
y2bey = lambda y: 2 *((1 + y/12) ** 6 - 1)

def pad_columns(tensor, target_cols, pad_mode='zero'):
    """
    Pads the columns of a 2D tensor to target_cols by adding padding values either
    as zeros or repeating the last value of each row.

    Args:
        tensor (torch.Tensor): Input tensor of shape (M, W)
        target_cols (int): Target number of columns after padding
        pad_mode (str): 'zero' to pad with zeros (default), 'last' to pad with last value per row

    Returns:
        torch.Tensor: Padded tensor of shape (M, target_cols)
    """
    M, W = tensor.shape
    C = target_cols - W
    if C <= 0:
        return tensor[:, :target_cols]  # truncate if needed
    
    if pad_mode == 'zero':
        pad = torch.zeros(M, C, dtype=tensor.dtype, device=tensor.device)
    elif pad_mode == 'last':
        last_vals = tensor[:, -1].unsqueeze(1)
        pad = last_vals.repeat(1, C)
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}. Use 'zero' or 'last'.")

    return torch.cat([tensor, pad], dim=1)

def pad_recovery_lag(v: torch.Tensor, recovery_lag: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Pads each 'row' along dimension 1 (WL) by different left padding amounts given in recovery_lag per batch item.

    Args:
        v (torch.Tensor): Input tensor of shape (M, WL, N)
        recovery_lag (torch.Tensor): Tensor of shape (M, 1) or (M,), with non-negative integers for left-pad size per batch
        max_len (int, optional): Output length along dim=1. If None, defaults to WL

    Returns:
        torch.Tensor: Output tensor of shape (M, WL, N), with zeros left-padded per batch according to recovery_lag.
    """
    M, WL, N = v.shape
    recovery_lag = recovery_lag.view(-1).type(torch.int64)  # shape (M,)
    assert recovery_lag.shape[0] == M, "recovery_lag should have the same batch size as v"

    if max_len is None:
        max_len = WL

    out = torch.zeros((M, max_len, N), dtype=v.dtype, device=v.device)

    # Source indices along dim=1 (length dimension)
    src_idx = torch.arange(WL, device=v.device).unsqueeze(0).expand(M, WL)  # shape (M, WL)
    # Destination indices = lag + source indices
    tgt_idx = src_idx + recovery_lag.unsqueeze(1)  # shape (M, WL)

    # Mask valid target positions (within max_len)
    valid_mask = tgt_idx < max_len

    # Batch indices for advanced indexing
    batch_idx = torch.arange(M, device=v.device).unsqueeze(1).expand(M, WL)  # (M, WL)

    # Use only valid positions for assignment
    batch_idx_valid = batch_idx[valid_mask]
    tgt_idx_valid = tgt_idx[valid_mask]
    src_idx_valid = src_idx[valid_mask]

    out[batch_idx_valid, tgt_idx_valid, :] = v[batch_idx_valid, src_idx_valid, :]
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
    netcfV: torch.Tensor
    totalDefaultV: torch.Tensor


class LoanAmort(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def agg_tensor(result_tensor):
        # Aggregate all members except 'months' by summing across dim=2 (loans), keep dim
        agg_fields = {}
        for field in result_tensor.__dataclass_fields__:
            val = getattr(result_tensor, field)
            if field == "months":
                agg_fields[field] = val
            else:
                agg_fields[field] = val.sum(dim=2, keepdim=True)
        return ResultTensor(**agg_fields)

    def forward(
            self, config, orig_wacV, wam, pv, rate_redV, refund_smmV, smmV, dqV, mdrV, sevV,
            aggMDR_timingV, aggMDR, compIntHC, servicing_fee, recovery_lag, 
            refund_premium, dq_adv_prin, dq_adv_int):
        """
        Args:
            wacV: [W, N]
            wam: [1, N]
            pv: [1, N]
            rate_redV: [M, W]
            refund_smmV, smmV, dqV, mdrV, sevV, aggMDR_timingV: [M, W]
            aggMDR, compIntHC, servicing_fee, recovery_lag, refund_premium, dq_adv_prin, dq_adv_int: [M, 1]

        Returns: result_tensor [N_loans, M_scenarios, WL_max_wam+lag, n_features]
        """

        N = wam.shape[1]
        M = smmV.shape[0]
        W = torch.max(wam)
        WL = int(torch.max(wam) + torch.max(recovery_lag))

        # pad tensors with shape W to WL (max wam + max recovery_lag) with zeros
        if WL != W:
            wacV = torch.cat([orig_wacV, torch.zeros(WL - W, N)], dim=0) # [WL, N]
            rate_redV = pad_columns(rate_redV, WL) # [M, WL]
            refund_smmV = pad_columns(refund_smmV, WL) # [M, WL]
            smmV = pad_columns(smmV, WL) # [M, WL]
            dqV = pad_columns(dqV, WL) # [M, WL]
            mdrV = pad_columns(mdrV, WL) # [M, WL]
            sevV = pad_columns(sevV, WL, pad_mode='last') # [M, WL]
            aggMDR_timingV = pad_columns(aggMDR_timingV, WL) # [M, WL]
        else:
            wacV = orig_wacV
        wacV = wacV.unsqueeze(0) - rate_redV.unsqueeze(2) # [M, WL, N]
        rateV = wacV / 12

        # Unfixed rate amortization
        exponent= torch.clamp(wam - torch.arange(WL).unsqueeze(1), min=1) # [M, N, WL], clamp 1 to avoid 0 in denominator for alpha
        alpha = rateV / ((1+rateV)**exponent - 1) 
        alpha = torch.cat([torch.zeros(M, 1, N), alpha], dim=1) # [M, WL+1, N], concat 0 to dimension 1
        balancesV = torch.nan_to_num(pv.repeat(WL+1,1) * torch.cumprod(1-alpha, dim=1)) # [M, WL+1, N]
        balancesV[torch.abs(balancesV) <= 1e-12] = 0.0
        principalsV = balancesV[:, :-1] - balancesV[:, 1:] # [M, WL, N]
        interestsV = balancesV[:, :-1] * rateV # [M, WL, N]
        paydownV = torch.nan_to_num(principalsV / balancesV[:, :-1]) # [M, WL, N]

        p_survV = torch.cumprod(1 - smmV - refund_smmV - mdrV, dim=1) # [M, WL]
        default_aggMDRV = pv.view(1, 1, N) * aggMDR.view(M, 1, 1) * aggMDR_timingV.view(M, WL, 1) # [M, WL, N]
        dqPrin_aggMDRV = paydownV * default_aggMDRV # [M, WL, N]
        scaled_default_aggMDRV = default_aggMDRV / (balancesV[:, :-1] * p_survV.unsqueeze(2) + 1e-16) # [M, WL, N]
        cum_scaled_default_aggMDRV = torch.cumsum(scaled_default_aggMDRV, dim=1) # [M, WL, N]
        survivorshipV = torch.cat([
            torch.ones(M, 1, N),
            p_survV.unsqueeze(2) * (1 - cum_scaled_default_aggMDRV)
        ], dim=1) # [M, WL+1, N]
        
        # Balances
        actualBalanceV = survivorshipV * balancesV # [M, WL+1, N]
        b_balanceV = actualBalanceV[:, :-1] # [M, WL, N], starts with month 0
        actualBalanceV = actualBalanceV[:, 1:] # [M, WL, N], starts with month 1

        # Scheduled Principals
        schedDQPrinV = survivorshipV[:, :-1] * principalsV * (1-mdrV.unsqueeze(2)) * dqV.unsqueeze(2) * (1-dq_adv_prin.unsqueeze(2)) # [M, WL, N]
        schedDefaultPrinV = survivorshipV[:, :-1] * principalsV * mdrV.unsqueeze(2) + dqPrin_aggMDRV # [M, WL, N]

        schedPrinV = survivorshipV[:, :-1] * principalsV - schedDQPrinV - schedDefaultPrinV # [M, WL, N]
        prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV.unsqueeze(2) # [M, WL, N]
        
        # Losses, recoveries, and writedowns
        defaultV = b_balanceV * mdrV.unsqueeze(2) + default_aggMDRV # [M, WL, N]
        writeDownV = pad_recovery_lag(defaultV, recovery_lag) # [M, WL, N]
        recoveryV = writeDownV * (1-sevV.unsqueeze(2)) # [M, WL, N]

        # Principals
        refundPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * refund_smmV.unsqueeze(2) # [M, WL, N]
        totalPrinV = schedPrinV + prepayPrinV + recoveryV # [M, WL, N]
        compIntV = prepayPrinV * rateV * compIntHC.unsqueeze(2) # [M, WL, N]
        refundIntV = refundPrinV * rateV # [M, WL, N]
        prepayPrinV = survivorshipV[:, :-1] * balancesV[:, 1:] * smmV.unsqueeze(2) + refundPrinV # [M, WL, N]

        # Servicing Fee
        defaultBalV = torch.clamp(torch.cumsum(defaultV - writeDownV, dim=1), min=0) # [M, WL, N]
        b_totalBalV = b_balanceV + torch.cat([torch.zeros(M, 1, N), defaultBalV[:, :-1]], dim=1) # [M, WL, N]
        totalBalV = actualBalanceV + defaultBalV # [M, WL, N]
        servicingFee_rate = servicing_fee.unsqueeze(2) / 12 # [M, 1]
        servicingFee_begV = b_totalBalV * servicingFee_rate # [M, WL, N]
        servicingFee_avgV = ((b_totalBalV + totalBalV) / 2) * servicingFee_rate # [M, WL, N]
        servicingFeeV = servicingFee_avgV if config.servicing_fee_method == 'avg' else servicingFee_begV

        actInterestV = rateV * b_balanceV if config.is_advance else (
            rateV * (b_balanceV * (1 - (1-mdrV.unsqueeze(2)) * dqV.unsqueeze(2) * (1-dq_adv_int.unsqueeze(2)) - mdrV.unsqueeze(2)) - default_aggMDRV) - compIntV)
        actInterestV -= refundIntV # [M, WL, N]

        cfV = totalPrinV + actInterestV # [M, WL, N]
        netCfV = torch.clamp(cfV - servicingFeeV, min=0) # [M, WL, N]
        totalDefaultV = schedDQPrinV + schedDefaultPrinV + defaultV # [M, WL, N]

        months = torch.arange(1, WL+1).unsqueeze(1).expand(M, WL, N) # [M, WL, N]

        if config.mode == "matched":
            # For each tensor of shape [M, WL, N], select [i, :, i] for i in range(N)
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
        M, WL, N = self.result_tensor.cfV.shape
        yV = (1 + y/12)**torch.arange(1, WL + 1) # [M, WL]
        if N == 1 and self.config.mode != "matched":
            pv = torch.sum(self.pv).expand(M, N, 1)
        elif self.config.mode == "matched":
            pv = self.pv.reshape(2, 1, 1)
        else:
            pv = self.pv.unsqueeze(1).expand(M, 1, N)
        px = torch.sum((self.result_tensor.cfV - self.result_tensor.servicingFeeV) / yV.unsqueeze(2), dim=1, keepdim=True) / (pv - torch.sum(self.result_tensor.refundPrinV / yV.unsqueeze(2), dim=1, keepdim=True)) # [M, 1, N]
        return px # [M, 1, N] price tensor
    
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
                pv_ = float(np.sum(pv)) if m_scenarios == 1 else float(pv[0, j].item())
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

        return torch.tensor(yield_tensor, dtype=torch.float64, device=result_tensor.cfV.device)

    def accr_int(wacV, rate_redV, settle_day):
        wacV = wacV.unsqueeze(0) - rate_redV.unsqueeze(2)
        return wacV * (settle_day - 1) / 360
