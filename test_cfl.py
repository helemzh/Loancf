import pytest
import numpy as np
from main import Loan, Yield, Output, Scenario

def test_py():
    loan = Loan(wac=0.0632, wam=357, pv=100000000)
    # Define vectors for the scenario
    smmVec = np.ones(loan.wam) * 0.01
    dqVec = np.ones(loan.wam) * 0.1
    mdrVec = np.ones(loan.wam) * 0.1
    sevVec = np.ones(loan.wam) * 0.2


    scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec)
    y = Yield(yieldValue=0.055)
    x = Yield(yieldValue=0.0632)

    output = Output(loan=loan, scenario=scenario, px=y)
    output1 = Output(loan=loan, scenario=scenario, px=x)

    df = output.getCashflow()
    px = output.getPX()

    assert np.isclose(px, 0.8281593283862828, rtol=0, atol=1e-8)