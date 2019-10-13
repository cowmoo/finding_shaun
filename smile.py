from connection import Connection
from py_vollib.black_scholes.greeks import analytical
from py_vollib.black_scholes import implied_volatility, black_scholes
import pickle
import datetime
import os
from ib_insync import *
import numpy as np
import math


class Cache:
    instance = None

    @classmethod
    def singleton(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        self.cache_path = "/Users/paulcao/option_chain_pickle.b"
        self.cache = pickle.load(open(self.cache_path, "rb")) if os.path.isfile(self.cache_path) else {}

    def commit(self):
        pickle.dump(self.cache, open(self.cache_path, "wb"))

    @staticmethod
    def keyForChain(contract, expiry):
        return ",".join([str(contract.conId), str(expiry)])

    def getChain(self, contract, expiry):
        key = Cache.keyForChain(contract, expiry)
        res = self.cache[key] if key in self.cache else None
        return res["result"] if (res != None) and (res["datetime"] > datetime.datetime.now() - expiry_timedelta) else None

    def existsChain(self, contract, expiry):
        return self.getChain(contract, expiry) != None and len(self.getChain(contract, expiry)) > 0

    def addChain(self, contract, expiry, results):
        self.cache[Cache.keyForChain(contract, expiry)] = \
            { "datetime": datetime.datetime.now(), "result": results}
        self.commit()

    @staticmethod
    def keyForSnapshot(symbol, expiry):
        return ",".join([symbol, expiry, "smile"])

    def getSnapshot(self, symbol, expiry):
        key = Cache.keyForSnapshot(symbol, expiry)
        res = self.cache[key] if key in self.cache else None
        return res["result"] if (res != None) and (res["datetime"] > datetime.datetime.now() - expiry_timedelta) else None

    def existsSnapshot(self, symbol, expiry):
        return self.getSnapshot(symbol, expiry) != None and len(self.getSnapshot(symbol, expiry)) > 0

    def addSnapshot(self, symbol, expiry, results):
        self.cache[Cache.keyForSnapshot(symbol, expiry)] = \
            {"datetime": datetime.datetime.now(), "result": results}
        self.commit()

    @staticmethod
    def keyForTicker(contract):
        return ",".join([str(contract.conId), "ticker"])

    def getTicker(self, contract):
        key = Cache.keyForTicker(contract)
        res = self.cache[key] if key in self.cache else None
        return res["result"] if (res != None) and (res["datetime"] > datetime.datetime.now() - expiry_timedelta) else None

    def existsTicker(self, contract):
        return self.getTicker(contract) != None and len(self.getTicker(contract)) > 0

    def addTicker(self, contract, results):
        self.cache[Cache.keyForTicker(contract)] = \
            {"datetime": datetime.datetime.now(), "result": results}
        self.commit()


class Chain:

    def __init__(self, expiry, contract, underlying_price):
        cache = Cache.singleton()

        #if cache.existsChain(contract, expiry):
        #    self.chain = cache.getChain(contract, expiry)
        #else:
        chain = Connection().reqContractDetails(Option(contract.symbol, expiry, exchange="SMART", currency="USD"))
        cache.addChain(contract, expiry, chain)
        self.chain = chain

        underlying_price = underlying_price
        upper_price, lower_price = 1.00 * underlying_price, 0.60 * underlying_price

        #contract_details = filter(lambda d: lower_price < d.contract.strike < upper_price and
        #                                    d.contract.right == 'P', self.chain)
        self.chain = map(lambda c: c.contract, self.chain)

    def __getattr__(self, name):
        return getattr(self.chain, name)

    def __next__(self):
        return next(self.chain)

    def __iter__(self):
        return self


class GreekCalculation:

    def __init__(self, contract, underlying_price, r=0.025, price=None, iv=None, bid=None, ask=None, bidSize=None, askSize=None):
        self.strike = contract.strike
        self.right = contract.right.lower()
        self.bid = bid
        self.bidSize = bidSize
        self.ask = ask
        self.askSize = askSize
        self.dt = (datetime.datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")
                   .replace(hour=16, minute=0) - datetime.datetime.now()).total_seconds() / (252 * 24 * 60 * 60)
        self.iv = iv if iv is not None else \
            GreekCalculation.implied_volatility(price, underlying_price, self.strike, self.dt, r, self.right)
        self.price = price if price is not None else \
            black_scholes(self.right, underlying_price, self.strike, self.dt, r, iv)
        self.delta = analytical.delta(self.right, underlying_price, float(self.strike), self.dt, r, self.iv)
        self.theta = analytical.theta(self.right, underlying_price, float(self.strike), self.dt, r, self.iv)
        self.vega = analytical.vega(self.right, underlying_price, float(self.strike), self.dt, r, self.iv)
        self.gamma = analytical.gamma(self.right, underlying_price, float(self.strike), self.dt, r, self.iv)

        if math.isnan(self.gamma) and self.delta == 0.0: # set gamma to 0 if delta is hovering at 0.0
            self.gamma = 0

        self.contract = contract

        prices = underlying_price * np.linspace(0.92, 1.06, 10)
        self.margin_impacts = []
        # np.zeros((10, ))

        # for i, iv in enumerate([0.095, 0.2425]):
        for i, iv in enumerate([0.2425]):
            for j, stress_price in enumerate(prices):
                self.margin_impacts.append(self.calcPrice(stress_price, iv) * float(self.contract.multiplier))

    @staticmethod
    def implied_volatility(price, underlying_price, strike, dt, r, right):
        try:
            return implied_volatility.implied_volatility(price, underlying_price, float(strike), dt, r, right)
        except Exception:
            return 0.0

    def calcPrice(self, underlying_price, iv, r=0.025):
        return black_scholes(self.right, underlying_price, self.strike, self.dt, r, iv)

    def calcVega(self, underlying_price, r=0.025):
        return analytical.vega(self.right, underlying_price, self.strike, self.dt, r, self.iv)


class Smile:

    def __init__(self, expiry, contract, underlying_price):
        #chain = list(Chain(expiry, contract, underlying_price))
        option_tickers = Connection().reqTickers(*list(Chain(expiry, contract, underlying_price)))
        self.smile = map(lambda c: GreekCalculation(c.contract, underlying_price, price=c.marketPrice(),
                                                    bid=c.bid, ask=c.ask), option_tickers)

    def __getattr__(self, name):
        return getattr(self.smile, name)

    def __next__(self):
        return next(self.smile)

    def __iter__(self):
        return self

