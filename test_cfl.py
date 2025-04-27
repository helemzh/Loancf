import pytest
import numpy as np
from main import Loan, Yield, Output, Scenario

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
        recovery_lagValue = 0
        aggMDR_Value = 0.0
 
        aggMDR_timing_Vec = np.zeros(loan.wam, dtype=float)
             
        scenario = Scenario(smmV=smm, dqV=mdr, mdrV=mdr, sevV=sev, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)   
        
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
    #test wal
    loan = Loan(wac=0.0632, wam=357, pv=100000000)

    # Test cases: (smm, mdr, sev, yield, expected_px)
    test_cases = [
     
        (0, 0, 0, 0.0632, 1.0000000000, 19.3142148090, 19.3142148090, 11.3140631626, 14.9166666667),
        (0.01, 0, 0, 0.0632, 1.0000000000, 6.9859506262, 6.9859506262, 6.2378547017, 6.7568208173),
        (0.01, 0.1, 0, 0.0632, 0.9954659556, 0.7514701165, 0.7514701165, 0.7511465462, 0.7514568528),
        (0.01, 0.1, 0.2, 0.0632, 0.8232870530, 0.7515413139, 0.7514701165, 0.7511465462, 0.7515217476)

        # Add more test cases here
    ]
    recovery_lagValue = 0
    aggMDR_Value = 0.0
 
    aggMDR_timing_Vec = np.zeros(loan.wam, dtype=float)
    
    for smm, mdr, sev, yield_value, expected_px, expected_wal_PrinV, expected_wal_BalanceDiffV, expected_wal_InterestV, expected_wal_cfl  in test_cases:
        dqV = mdr
        # Create scenario
        #scenario = Scenario(smm, mdr, sev, 0)  # Assuming DQ is always 0
     
    
     
        scenario = Scenario(smmV=smm, dqV=mdr, mdrV=mdr, sevV=sev, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)
        
        
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
def test_py3():
    # Test balance after introducing a new aggmdr
    loan = Loan(wac=0.06, wam=24, pv=100000)

    # Test cases: (smm, mdr, sev, yield, expected_px)
    test_cases = [
     
        (0.01, 0, 0, 0.0632, 94107.2595, 88333.8282, 82680.1672)
     

        # Add more test cases here
    ]

     
    aggMDR_Value = 0.1
    x = np.zeros(loan.wam, dtype=float)
    np.full_like(x, 0.01)
    aggMDR_timing_Vec = np.full_like(x, 0.1)
 
   
 #   aggMDR_timingv = np.ones(loan.wam) * 0.0
    #recovery_lagValue = 2
    recovery_lagValue = 0
    for smm, mdr, sev, yield_value, expected_actualbal0, expected_actualbal1, expected_actualbal2 in test_cases:
        
        smmVec = np.ones(loan.wam) * 0.01
        mdrVec = np.ones(loan.wam) * mdr
        
        dqV = mdrVec
       
        scenario = Scenario(smmV=smmVec, dqV=mdrVec, mdrV=mdrVec, sevV=sev, recovery_lag=recovery_lagValue, aggMDR=aggMDR_Value, aggMDR_timingV= aggMDR_timing_Vec)
        
        # Create yield object
        y = Yield(yieldValue=yield_value)
        
        # Create output object
        output = Output(loan=loan, scenario=scenario, px=y)
        
        df = output.getCashflow()
        # Get price
        px = output.getPX()

        #wal_PrinV = output.get_wal_PrinV()
        #wal_BalanceDiffV = output.get_wal_BalanceDiffV()
        #wal_InterestV = output.get_wal_InterestV()
        #wal_cfl = output.get_wal_cfl()  

        actualbal0 = output.get_actualbal0()
        actualbal1 = output.get_actualbal1()
        actualbal2 = output.get_actualbal2()   
        # Assert
    #    assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
    #        f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"
        
      
        assert np.isclose(actualbal0, expected_actualbal0, rtol=0, atol=1e-4)
        assert np.isclose(actualbal1, expected_actualbal1, rtol=0, atol=1e-4)
        assert np.isclose(actualbal2, expected_actualbal2, rtol=0, atol=1e-4)
        
        # assert without wal
        # assert np.isclose(px, expected_px, rtol=0, atol=1e-7), \
        #    f"Failed for smm={smm}, mdr={mdr}, sev={sev}, yield={yield_value}. Expected {expected_px}, got {px}"

        print("All tests passed successfully!")      



if __name__ == '__main__':
    test_py1()
    test_py2()
    test_py3()
