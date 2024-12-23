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
    #smm=0_mdr=0_sev=0 monnthly y=.055;px=1.0919218749 
    #smm=0_mdr=0_sev=0 monnthly y=.0632;px=1.0000000000 
    #scenario = Scenario(0, 0, 0, 0)   
    #smm=0.01_mdr=0_sev=0 monnthly y=.055;px=1.0423018980
    #scenario = Scenario(0.01, 0, 0, 0)    
    y = Yield(yieldValue=0.055)
    #y = Yield(yieldValue=0.0632)
   # x = Yield(yieldValue=0.0632)

    output = Output(loan=loan, scenario=scenario, px=y)
  #  output1 = Output(loan=loan, scenario=scenario, px=x)

    df = output.getCashflow()
    px = output.getPX()

    assert np.isclose(px, 0.8281593283862828, rtol=0, atol=1e-8)
    #assert np.isclose(px, 1.0919218749, rtol=0, atol=1e-7)
    #assert np.isclose(px, 1.0000000000, rtol=0, atol=1e-7)
    #assert np.isclose(px, 1.0423018980, rtol=0, atol=1e-7)

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
        dqV = mdr
        # Create scenario
        #scenario = Scenario(smm, mdr, sev, 0)  # Assuming DQ is always 0
        scenario = Scenario(smm, dqV, mdr, sev)
        
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

if __name__ == '__main__':
    test_py1()
