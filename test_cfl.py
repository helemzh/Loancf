import pytest
import numpy as np
from main import Loan, Yield, Output, Scenario

def test_py():
    loan = Loan(wac=0.0632, wam=357, pv=100000000)
    # Define vectors for the scenario 3
    smmVec = np.ones(loan.wam) * 0.01
    dqVec = np.ones(loan.wam) * 0.1
    mdrVec = np.ones(loan.wam) * 0.1
    sevVec = np.ones(loan.wam) * 0.0

     # Define vectors for the scenario 2
    # smmVec = np.ones(loan.wam) * 0.01
    #dqVec = np.ones(loan.wam) * 0.0
    # mdrVec = np.ones(loan.wam) * 0.0
    # sevVec = np.ones(loan.wam) * 0.0  

  # Define vectors for the scenario 1
  # smmVec = np.ones(loan.wam) * 0.0
  #  dqVec = np.ones(loan.wam) * 0.0
  #  mdrVec = np.ones(loan.wam) * 0.0
  #  sevVec = np.ones(loan.wam) * 0.0

    # Define vectors for the scenario 4
    #smmVec = np.ones(loan.wam) * 0.01
    #dqVec = np.ones(loan.wam) * 0.1
    #mdrVec = np.ones(loan.wam) * 0.1
    #sevVec = np.ones(loan.wam) * 0.2

    scenario = Scenario(smmV=smmVec, dqV=dqVec, mdrV=mdrVec, sevV=sevVec)
    #smm=0_mdr=0_sev=0 monnthly y=.055;px=1.0919218749 
    #smm=0_mdr=0_sev=0 monnthly y=.0632;px=1.0000000000 
    #scenario = Scenario(0, 0, 0, 0)   
    #smm=0.01_mdr=0_sev=0 monnthly y=.055;px=1.0423018980
    #scenario = Scenario(0.01, 0, 0, 0)    
    #y = Yield(yieldValue=0.055)
    y = Yield(yieldValue=.0632)
    #y = Yield(yieldValue=0.0632)
   # x = Yield(yieldValue=0.0632)

    output = Output(loan=loan, scenario=scenario, px=y)
  #  output1 = Output(loan=loan, scenario=scenario, px=x)

    df = output.getCashflow()
    px = output.getPX()
    wal_PrinV = output.get_wal_PrinV()
    wal_BalanceDiffV = output.get_wal_BalanceDiffV()
    wal_InterestV = output.get_wal_InterestV()
    wal_cfl = output.get_wal_cfl()

   

    # assert for scenario 3
    assert np.isclose(px, 0.9954659556, rtol=0, atol=1e-8)
    assert np.isclose(wal_PrinV, 0.7514701165, rtol=0, atol=1e-8)
    assert np.isclose(wal_BalanceDiffV, 0.7514701165, rtol=0, atol=1e-8)
    assert np.isclose(wal_InterestV, 0.7511465462, rtol=0, atol=1e-8)
    assert np.isclose(wal_cfl, 0.7514568528, rtol=0, atol=1e-8)

    # assert for scenario 2
    # assert np.isclose(px, 1.0000000000, rtol=0, atol=1e-8)
    # assert np.isclose(wal_PrinV, 6.9859506262, rtol=0, atol=1e-8)
    # assert np.isclose(wal_BalanceDiffV, 6.9859506262, rtol=0, atol=1e-8)
    # assert np.isclose(wal_InterestV, 6.2378547017, rtol=0, atol=1e-8)
    # assert np.isclose(wal_cfl, 6.7568208173, rtol=0, atol=1e-8)   

 # assert for scenario 1
   # assert np.isclose(px, 1.0000000000, rtol=0, atol=1e-8)
   # assert np.isclose(wal_PrinV, 19.3142148090, rtol=0, atol=1e-8)
   # assert np.isclose(wal_BalanceDiffV, 19.3142148090, rtol=0, atol=1e-8)
   # assert np.isclose(wal_InterestV, 11.31406316262, rtol=0, atol=1e-8)
   # assert np.isclose(wal_cfl, 14.91666666676, rtol=0, atol=1e-8)

    # assert for scenario 4
    #assert np.isclose(px, 0.8232870530, rtol=0, atol=1e-8)
    #assert np.isclose(wal_PrinV, 0.7515413139, rtol=0, atol=1e-8)
    #assert np.isclose(wal_InterestV, 0.7511465462, rtol=0, atol=1e-8)
    #assert np.isclose(wal_BalanceDiffV, 0.7514701165, rtol=0, atol=1e-8)
    #assert np.isclose(wal_cfl, 0.7515217476, rtol=0, atol=1e-8)
   
    #assert np.isclose(px, 0.8281593283862828, rtol=0, atol=1e-8)
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

def test_py2():
    #add wal 
    loan = Loan(wac=0.0632, wam=357, pv=100000000)

    # Test cases: (smm, mdr, sev, yield, expected_px)
    test_cases = [
     
        (0, 0, 0, 0.0632, 1.0000000000, 19.3142148090, 19.3142148090, 11.3140631626, 14.9166666667),
        (0.01, 0, 0, 0.0632, 1.0000000000, 6.9859506262, 6.9859506262, 6.2378547017, 6.7568208173),
        (0.01, 0.1, 0, 0.0632, 0.9954659556, 0.7514701165, 0.7514701165, 0.7511465462, 0.7514568528),
        (0.01, 0.1, 0.2, 0.0632, 0.8232870530, 0.7515413139, 0.7514701165, 0.7511465462, 0.7515217476)

        # Add more test cases here
    ]

    for smm, mdr, sev, yield_value, expected_px, expected_wal_PrinV, expected_wal_BalanceDiffV, expected_wal_InterestV, expected_wal_cfl  in test_cases:
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

    print("All tests passed successfully!")   

if __name__ == '__main__':
  #  test_py1()
  #  test_py()
    test_py2()
