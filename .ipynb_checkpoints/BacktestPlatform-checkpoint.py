import pandas as pd
import numpy as np
class TradeBlotter:
    def __init__(self):
        self.entryDate = []
        self.entryPrice = []
        self.direction = []
        self.exitPrice = []
        self.live = []
        self.market = []
        self.realizedPnl = []
        self.stopLevel = []
        self.stopOrExit = []
        self.unrealizedPnl = []
        self.exitDate = []
        self.portfolioHistory = {}
        self.openTrades = {}
        self.lastPortfolioUpdateDate = None
        
    def toDataFrame(self):
        return pd.DataFrame(np.array([self.entryDate, self.market, self.entryPrice, self.direction, self.live, self.stopLevel, self.exitPrice, self.exitDate, self.unrealizedPnl, self.realizedPnl, self.stopOrExit]).T,
                           columns=['Entry Date', 'Market', 'Entry Price', 'Direction', 'Live', 'Stop Level', 'Exit Price', 'Exit Date', 'Unrealized PnL', 'Realized PnL', 'Stop/Exit'])
        
class Details:
    def __init__(self, unit, entryPrice, entryDate):
        self.unit = unit
        self.entryPrice = entryPrice
        self.entryDate = entryDate
        self.unrealizedPnl = 0
        
    def toList(self):
        return [self.entryPrice, self.entryDate, self.unit, self.unrealizedPnl]
    
    