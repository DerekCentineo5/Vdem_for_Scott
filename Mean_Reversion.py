import numpy as np
from numpy import linalg
from datetime import date, datetime, timedelta
import scipy.optimize as opt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

class Portfolio():

    def __init__(self, Data_Table, Tickers):
        """
        Data_Table: DataFrame of Assets with their Close Price Data
        Tickers: (list) Tickers and/or Names of assets.

        """
        self.Tickers = Tickers
        self.Prices = Data_Table
        self.Start = self.Prices.index[0]
        self.End = self.Prices.index[-1]

        self.initial_weights = np.ones(len(self.Tickers)) / len(self.Tickers)

        self.EG_weights = [np.ones(len(self.Tickers)) / len(self.Tickers)]
    
    def Mean_Reversion(self, epsilon=5, alpha=np.arange(0, 1, .1)[1:], eta=10):
        """

        Parameters
        -----------
            epsilon: (float) Reversion threshold with range [1, inf). 
            alpha: (float) Exponential ratio for the Exponentially Weighted Average with range (0,1).\\
            --> Larger Alpha = More Importance on Recent Prices\\
            --> Smaller Alpha = More Importance on Historical Prices
            eta: (float) "Learning Rate AKA Regulator"

        """
        ### Vars ###
        Prices = self.Prices
        self.eta = eta
        self.alpha = alpha
        self.Price_Relative_EWMA = np.zeros((len(self.alpha), len(Prices.columns)))
        self.Losses = np.zeros(len(self.alpha))
        self.epsilon = epsilon
        ### Vars ###

        Price_Rels = np.array((Prices).pct_change().fillna(0) + 1)

        Total_Time = np.array(Prices.index)

        ### Create Portfolio through our iterations ###

        for iteration in range(len(Total_Time)):

            Price_Data = Price_Rels[:iteration+1]

            T, N = Price_Data.shape

            for i, a in enumerate(self.alpha):

                self.Losses[i] += np.mean((Price_Data[-1] - self.Price_Relative_EWMA[i])**2)

                EWMA = a + (1 - a) * self.Price_Relative_EWMA[i] / Price_Data[-1]

                self.Price_Relative_EWMA[i] = EWMA 
            
            Weights = -self.eta * self.Losses

            Weights = np.exp(Weights) / np.sum(np.exp(Weights))

            Relatives = self.Price_Relative_EWMA.T.dot(Weights)

            Loss_Func = max(0, (self.epsilon - np.dot(self.EG_weights[-1],Relatives)))

            Mean_Rel = np.mean(Relatives)

            Denominator = (np.dot((Relatives - Mean_Rel),(Relatives - Mean_Rel)))

            if Denominator != 0:
                Lamd = Loss_Func / Denominator
            
            else:
                Lamd = 0
            
            new_weight = self.EG_weights[-1] + Lamd*(Relatives - Mean_Rel)

            Desc = np.sort(new_weight)[::-1] #Descending Order

            adjusted_sum = np.cumsum(Desc) - 1 

            j = np.arange(len(new_weight)) + 1

            cond = Desc - adjusted_sum / j > 0

            # If conditions are not met, we go to equal weights.

            if not cond.any():

                final_weight = np.ones(len(new_weight)) / len(new_weight)
            
            else:
               
                rho = float(j[cond][-1])
                
                theta = adjusted_sum[cond][-1] / rho

                final_weight = np.maximum(new_weight - theta, 0)
            
            self.EG_weights.append(final_weight)
        
        Historical_Weights = pd.DataFrame(self.EG_weights[:-1], columns=Prices.columns, index=Prices.index)

        Predicted_Weights = pd.DataFrame(self.EG_weights[-1:], columns=Prices.columns, index=(Historical_Weights.index[-1:] + BDay(1)))
        
        return Historical_Weights, Predicted_Weights
           