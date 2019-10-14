import asyncio
import nest_asyncio
from connection import Connection
from ib_insync import *
from smile import Smile
import pickle
import math
import cvxpy as cp
import cvxpy.atoms.affine.sum as cx
import datetime


async def run_scanner():

    pairs = [ETFPair(1, 3, "GDX", "NUGT", n_expiry=1, call=True),
             ETFPair(1, 3, "XLF", "FAS", n_expiry=1, call=True),
             ETFPair(1, 3, "IWM", "TNA", n_expiry=1, call=True),
             ETFPair(2, 3, "VXX", "UVXY", n_expiry=1, call=True)]
    for pair in pairs: await pair.solve()

    def display_spread(spread):
        print("Spread: " + str(spread.spread))
        print("Unlevered: " + str(spread.unlevered_contract.contract) + ' @ $' + str(spread.unlevered_contract.price) \
                + ' x' + str(spread.unlevered_real_ratio))

        unlevered_spot_change = (spread.unlevered_contract.contract.strike - spread.prices[0].marketPrice()) \
                      / spread.prices[0].marketPrice()
        print("Unlevered Spot: " + str(spread.prices[0].marketPrice()) + "; Pct. Change to Touch:  " + str(unlevered_spot_change))

        print("Levered: " + str(spread.levered_contract.contract) + ' @ $' + str(spread.levered_contract.price) \
                + ' x-' + str(spread.levered_real_ratio))

        levered_spot_change = (spread.levered_contract.contract.strike - spread.prices[1].marketPrice()) \
                      / spread.prices[1].marketPrice()
        print("Levered Spot: " + str(spread.prices[1].marketPrice()) + "; Pct. Change to Touch: " + str(levered_spot_change))
        print("\n")

    for pair in list(sorted(pairs, key=lambda p: p.spread, reverse=True)):
        display_spread(pair)


def generate_expiry(n):
    def next_expiry(dt, i, cboe_listed=False):
        dt = dt + datetime.timedelta(days=7*i)
        friday = dt + datetime.timedelta((4 - dt.weekday()) % 7)
        thursday = dt + datetime.timedelta((3 - dt.weekday()) % 7)
        return thursday if cboe_listed and 15 <= thursday.day <= 21 else friday

    expiries_dt = map(lambda i: next_expiry(datetime.date.today(), i), range(n))
    return list(map(lambda dt: dt.strftime('%Y%m%d'), expiries_dt))


class ETFPair:

    def __init__(self, unlevered_ratio, levered_ratio, unlevered_symbol, levered_symbol, n_expiry=3, call=True):
        self.unlevered_ratio = unlevered_ratio
        self.levered_ratio = levered_ratio
        self.expiries = generate_expiry(n_expiry)
        self.unlevered_symbol, self.levered_symbol, self.call = unlevered_symbol, levered_symbol, call

    async def solve(self):
        self.contracts = Connection().qualifyContracts(*[Stock(self.unlevered_symbol, exchange='SMART', currency='USD'),
                                Stock(self.levered_symbol, exchange='SMART', currency='USD')])
        self.prices = Connection().reqTickers(*self.contracts)

        pairs = list(map(lambda expiry: [list(Smile(expiry, self.contracts[0], self.prices[0].marketPrice())),
                            list(Smile(expiry, self.contracts[1], self.prices[1].marketPrice()))], self.expiries))
        arbs = list(map(lambda pair: ArbSmile(self.unlevered_ratio, self.levered_ratio, pair[0], pair[1],
                                              self.prices[0].marketPrice(), self.prices[1].marketPrice(), call=self.call), pairs))
        best = sorted(arbs, key=lambda arb: arb.ans, reverse=True)[0]
        self.unlevered_contract, self.levered_contract, self.spread, self.unlevered_real_ratio, self.levered_real_ratio = \
            best.unlevered_contract, best.levered_contract, best.ans, best.standardized_lratio, best.unlevered_ratio


class ArbSmile:

    @staticmethod
    def filter(smile, spot, call):
        side = 'c' if call else 'p'
        return list(filter(lambda g: g.right == side and g.strike > spot, smile)) if call \
            else list(filter(lambda g: g.right == side and g.strike < spot, smile))

    @staticmethod
    def idx(matrix): return int(next(filter(lambda v: v[1] > 0, enumerate(matrix.value)))[0])

    def __init__(self, unlevered_ratio, levered_ratio, unlevered_smile, levered_smile, unlevered_spot, levered_spot,
                 safety_margin=1, call=True):
        self.unlevered_ratio, self.levered_ratio, self.unlevered_spot, self.levered_spot, self.safety_margin = \
            unlevered_ratio, levered_ratio, unlevered_spot, levered_spot, safety_margin
        self.unlevered_smile = ArbSmile.filter(unlevered_smile, unlevered_spot, call)
        self.levered_smile = ArbSmile.filter(levered_smile, levered_spot, call)
        self.ans, self.levered_contract, self.unlevered_contract = self.solve()

    def solve(self):
        self.levered_smile = list(filter(lambda g: not math.isnan(g.price), sorted(self.levered_smile,
                                                                               key=lambda g: g.strike)))
        self.unlevered_smile = list(filter(lambda g: not math.isnan(g.price), sorted(self.unlevered_smile,
                                                                                key=lambda g:g.strike)))

        lstrikes, ulstrikes = list(map(lambda g: g.strike, self.levered_smile)), \
                              list(map(lambda g: g.strike, self.unlevered_smile))

        lprices, ulprices = list(map(lambda g: g.price, self.levered_smile)),\
                            list(map(lambda g: g.price, self.unlevered_smile))

        lpcts, ulpcts = list(map(lambda strike: (strike - self.levered_spot)/self.levered_spot, lstrikes)), \
                        list(map(lambda strike: (strike - self.unlevered_spot) / self.unlevered_spot, ulstrikes))

        ltheta, ultheta = list(map(lambda g: g.theta, self.levered_smile)),\
                            list(map(lambda g: g.theta, self.unlevered_smile))

        lcontract, ulcontract = cp.Variable(len(lstrikes), boolean=True),  cp.Variable(len(ulstrikes), boolean=True)

        spot_ratio = self.levered_spot / self.unlevered_spot
        self.standardized_lratio = round(self.levered_ratio * spot_ratio)

        max_theta = False
        objective = cx.sum(-self.unlevered_ratio * lcontract * ltheta + self.standardized_lratio * ulcontract * ultheta) \
            if max_theta else cx.sum(self.unlevered_ratio * lcontract * lprices - self.standardized_lratio * ulcontract * ulprices)

        prob = cp.Problem(100 * cp.Maximize(objective),
                          [cx.sum(lcontract) == 1,
                           cx.sum(ulcontract) == 1,
                           cx.sum(self.levered_ratio * ulcontract * ulpcts) <= cx.sum(self.unlevered_ratio * lcontract * lpcts),
                           cx.sum(self.safety_margin * self.standardized_lratio * ulcontract * ulpcts) <=
                            cx.sum(self.unlevered_ratio * lcontract * lpcts)])

        ans = prob.solve(solver=cp.GLPK_MI)

        return ans, self.levered_smile[ArbSmile.idx(lcontract)], self.unlevered_smile[ArbSmile.idx(ulcontract)]


if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    futures = [run_scanner()]
    loop.run_until_complete(asyncio.wait(futures))
    loop.close()