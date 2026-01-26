# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#%%

def fetchStockInfo(TICKER, printInfo = True, advancedInformation = False):
    """
    

    Parameters
    ----------
    TICKER : Str
        Company stock ticker name, enter in quotation marks. Make sure ticker is same as yahoo finance listing
        
    advancedInformation : Bool
        If true, will display more advanced company metrics like PE and fair pricing estimates.

    Returns
    -------
    latestPrice : Float
        The most recent known stock price.
    
    companyMC_str : String
        Company Market cap
        
    companyPE_str : string
        The price to earnings ratio of the company
        
    companyDiv_str : str
        The dividend yield paid out by the company annually
    """
    
    #fetch company ticker
    company = yf.Ticker(TICKER)
    
    #fetch ticker stock price
    data = company.history(period="1d", interval = "1m")
    currency = company.info.get("currency", "USD")
    
    #return none if incorrect ticker
    if data.empty:
     print(f"No data available for {TICKER}")
     return None
    
    latestPrice = data['Close'].iloc[-1]
    
    
    
    #print the stock price/share
    if printInfo == True:
        print(company.info.get("longName", "N/A"))
        print(f'The latest price of {TICKER} is: {currency}{latestPrice:,.2f}')
    
    companyMC = company.info.get("marketCap", "N/A")
    companyPE = company.info.get("forwardPE", "N/A")
    companyDiv = company.info.get("dividendRate", "N/A")
    
    companyMC_str = f"{currency}{companyMC:,}" if isinstance(companyMC, (int, float)) else "N/A"
    companyPE_str = f"{companyPE:.2f}" if isinstance(companyPE, (int, float)) else "N/A"
    companyDiv_str = f"{companyDiv:.2f}%" if isinstance(companyDiv, (int, float)) else "N/A"
    
    
    
    data = yf.Ticker(TICKER).history(period="1y")['Close']
    
    # daily returns
    returns = data.pct_change().dropna()
    
    dailyVolatility = np.std(returns)
    annualVol = dailyVolatility * np.sqrt(252)

    
    startPrice = data.iloc[0]
    endPrice = data.iloc[-1]
    years = len(data) / 252  
    annualReturn = (endPrice / startPrice) ** (1/years) - 1
        
    #display advanced information if desired
    if advancedInformation == True:
                 
        print(
            f"Market cap: {companyMC_str}\n"
            f"PE: {companyPE_str}\n"
            f"Dividend yield: {companyDiv_str}\n"
            f"Annualized volatility: {annualVol:2f}\n"
            f"Approx. annualized return: {annualReturn:.2%}"
        )
        
    return latestPrice, companyMC_str, companyPE_str, companyDiv_str, annualVol, annualReturn


#%%


def returnsCalculator(MonthlyDeposit, StartingBalance, EquityWeight, RiskFreeRateWeight, TICKER, RiskFreeRateReturn, NumberOfMonths = 24, NumberOfSimulations = 10, Leverage = False): 
    
    
    
    """ 
    
    Parameters 
    ---------- 
    MonthlyDeposit : Float 
        The monthly contributions to the portfolio
    
    StartingBalance : Float 
        The Starting value of the portfolio
    
    EquityWeight : Float 
        The weight of Equity in the portfolio. The value of Equities divided by the value of risk free return vehicle. Ensure Leverage is set to 1 if the sum of Equity weight and Risk free rate weight exceed 1
    
    RiskFreeRateWeight : Float 
        The weight of risk free return vehicles in the portfolio. The value of risk free return vehicle divided by the value of equities. Ensure Leverage is set to 1 if the sum of Equity weight and Risk free rate weight exceed 1
    
    Leverage : Bool 
    Set to True if you are using leverage. if you dont know what this is, please set to False or omit  (default = False)
    
    EquityReturn : Float 
        The hypothetical return of the equity investment in percentage terms
    
    EquityVariance : Float 
        The hypothetical variance of the equity investment
    
    RiskFreeRateReturn : Float 
        The return of the market rate in percentage terms
    
    NumberOfMonths : Integer 
        How many months would you like to simulate (default = 24)
        
    NumberOfSimulations : Integer 
        How many simulations would you like to compute (default = 10)
        
    Returns 
    ------- 
    returnsCalculator : Float 
        The Value of the portfolio after so many years
    
    """ 
    

    
    MonthlyDeposit = float(MonthlyDeposit)
    StartingBalance = float(StartingBalance) 
    totalWeight = float(RiskFreeRateWeight + EquityWeight)
    EquityReturn = fetchStockInfo(TICKER, False)[5]
    EquityVariance = fetchStockInfo(TICKER, False)[4]
    RiskFreeRateReturn = float(RiskFreeRateReturn)
    NumberOfMonths = int(NumberOfMonths) 
    
    if not isinstance(MonthlyDeposit, float): 
        raise ValueError(f'expected "Monthly Deposit" to be a float, got {type(MonthlyDeposit).__name__}') 
    
    if not isinstance(StartingBalance, float): 
        raise ValueError(f'expected "Starting Balance" to be a float, got {type(StartingBalance).__name__}') 
    
    if not 0<EquityWeight<1 and Leverage == True: 
        raise ValueError(f'expected "Equity Weight" to be a Float between 0 and 1, got {type(EquityWeight).__name__}') 
    
    if not 0<RiskFreeRateWeight<1 and Leverage == True: 
        raise ValueError(f'expected "Risk free rate weight" to be a Float between 0 and 1, got {type(RiskFreeRateWeight).__name__}') 
    
    if not totalWeight == 1 and Leverage == False: 
        raise ValueError('expected "Equity weight and Risk free rate weight" to sum to 1, but they do not, please ensure that leverage is enabled if your total exposure exceeds 1') 
    
    if not isinstance(EquityReturn, float): 
        raise ValueError(f'expected "Equity Return" to be a float, got {type(EquityReturn).__name__}') 
    
    if not isinstance(EquityVariance, float): 
        raise ValueError(f'expected "Equity Variance" to be a float, got {type(RiskFreeRateReturn).__name__}') 
    
    if not isinstance(RiskFreeRateReturn, float): 
        raise ValueError(f'expected "Risk free rate return" to be a float, got {type(RiskFreeRateReturn).__name__}') 
    
    if not isinstance(NumberOfMonths, int): 
        raise ValueError(f'expected "Number of Months" to be an integer, got {type(NumberOfMonths).__name__}')
        
    if not isinstance(NumberOfSimulations, int): 
        raise ValueError(f'expected "Number of Simulations" to be an integer, got {type(NumberOfSimulations).__name__}')

    plt.figure(figsize=(10,6))

    AllPaths = []



    for j in range(NumberOfSimulations):

        
        CurrentBalance = np.array([StartingBalance])
        
        for i in range(NumberOfMonths):
            
            NewBalance = round(( CurrentBalance[-1] + MonthlyDeposit ) + ( CurrentBalance[-1] * np.random.normal(EquityReturn/12, (EquityVariance/np.sqrt(12))) * EquityWeight ) + ( CurrentBalance[-1] * RiskFreeRateWeight * RiskFreeRateReturn/100 ), 2) 
            
            CurrentBalance = np.append(CurrentBalance, NewBalance)
            
            Months = np.arange(len(CurrentBalance))
        
        AllPaths.append(CurrentBalance)
            
    
    
    
    MeanPath = np.mean(AllPaths, axis=0)
    Months = np.arange(NumberOfMonths + 1)
    
    print("The Returns on this portfolio:", {MeanPath[-1] - StartingBalance - NumberOfMonths * MonthlyDeposit})
    
    plt.figure(figsize=(10,6))
    for i in AllPaths:
        plt.plot(Months, i, color='blue', linewidth = .25)
    plt.plot(Months, MeanPath, color='red', linewidth=2, label='Mean Portfolio')
    plt.plot([], [], color='blue', linewidth=0.75, label='Individual Simulations')
    
    plt.xlabel('Month')
    plt.ylabel('Portfolio Value')
    plt.title('Projected Portfolio Balance')
    plt.legend(fontsize = 12,  loc = "upper left")
    plt.grid(True)
    plt.show()
    

    
    return AllPaths, MeanPath
    
    # return CurrentBalance

        
            

        


#%%
stock = "AAPL"

returnsCalculator(500, 0, 1, 0, f"{stock}", 0)

fetchStockInfo(f"{stock}", True, True)












