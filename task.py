import asyncio
import nest_asyncio
from connection import Connection
from ib_insync import *
from smile import Smile
import pickle
import math
import cvxpy as cp
import cvxpy.atoms.affine.sum as cx


async def run_scanner():
    expiry = "20191025" #"20191101", "20191108", 20191115, 20191025, 20191018
    cache_path = "/Users/paulcao/arb_nugt.b"
    uvxy_stock = Connection().qualifyContracts(*[Stock('NUGT', exchange='SMART', currency='USD')])[0]
    uvxy_underlying_price = Connection().reqTickers(*[uvxy_stock])[0].marketPrice()
    print("UVXY: " + str(uvxy_underlying_price))

    vxx_stock = Connection().qualifyContracts(*[Stock('GDX', exchange='SMART', currency='USD')])[0]
    vxx_underlying_price = Connection().reqTickers(*[vxx_stock])[0].marketPrice()
    print("VXX: " + str(vxx_underlying_price))

    uvxy_smile = list(Smile(expiry, uvxy_stock, uvxy_underlying_price))
    print(uvxy_smile)

    vxx_smile = list(Smile(expiry, vxx_stock, vxx_underlying_price))
    print(vxx_smile)

    res = {"uvxy_underlying_price": uvxy_underlying_price,
           "vxx_underlying_price": vxx_underlying_price,
           "uvxy_smile": uvxy_smile,
           "vxx_smile": vxx_smile}
    pickle.dump(res, open(cache_path, "wb"))
    print("Committed into " + str(cache_path) + ".")


async def load_scanner():
    cache_path = "/Users/paulcao/arb_nugt.b"
    cache = pickle.load(open(cache_path, "rb"))

    print("UVXY: " + str(cache["uvxy_underlying_price"]))
    print("VXX: " + str(cache["vxx_underlying_price"]))

    #print(cache["uvxy_smile"])
    #print(cache["vxx_smile"])

    arbitrage = Arbitrage(1, 3, cache["vxx_smile"], cache["uvxy_smile"], cache["vxx_underlying_price"], cache["uvxy_underlying_price"])
    arbitrage.solve()


class Arbitrage:

    def __init__(self, unlevered_ratio, levered_ratio, unlevered_smile, levered_smile, unlevered_spot, levered_spot, side="c"):
        self.unlevered_ratio = unlevered_ratio
        self.levered_ratio = levered_ratio

        if side == "c":
            self.unlevered_smile = list(filter(lambda g: g.right == side and g.strike > unlevered_spot, unlevered_smile))
            self.levered_smile = list(filter(lambda g: g.right == side and g.strike > levered_spot, levered_smile))
        elif side == 'p':
            self.unlevered_smile = list(filter(lambda g: g.right == side and g.strike < unlevered_spot, unlevered_smile))
            self.levered_smile = list(filter(lambda g: g.right == side and g.strike < levered_spot, levered_smile))

        self.unlevered_smile = list(filter(lambda g: not math.isnan(g.price), self.unlevered_smile))
        self.unlevered_smile = list(sorted(self.unlevered_smile, key=lambda g: g.strike))
        self.levered_smile = list(filter(lambda g: not math.isnan(g.price), self.levered_smile))
        self.levered_smile = list(sorted(self.levered_smile, key=lambda g: g.strike))

        self.unlevered_spot = unlevered_spot
        self.levered_spot = levered_spot

    def solve(self):
        levered_strikes = list(map(lambda g: g.strike, self.levered_smile))
        unlevered_strikes = list(map(lambda g: g.strike, self.unlevered_smile))

        levered_prices = list(map(lambda g: g.price, self.levered_smile))
        unlevered_prices = list(map(lambda g: g.price, self.unlevered_smile))

        unlevered_pcts= list(map(lambda strike: (strike - self.unlevered_spot)/self.unlevered_spot, unlevered_strikes))
        levered_pcts = list(map(lambda strike: (strike - self.levered_spot)/self.levered_spot, levered_strikes))

        unlevered_leg = cp.Variable(len(unlevered_strikes), boolean=True)
        levered_leg = cp.Variable(len(levered_strikes), boolean=True)
        safety_margin = 1

        prob = cp.Problem(
            cp.Maximize(100 * cx.sum(self.unlevered_ratio * levered_leg * levered_prices - self.levered_ratio * unlevered_leg * unlevered_prices)), [
                cx.sum(levered_leg) == 1,
                cx.sum(unlevered_leg) == 1,
                cx.sum(safety_margin * self.levered_ratio * unlevered_leg * unlevered_pcts) <=
                        cx.sum(self.unlevered_ratio * levered_leg * levered_pcts)
                #have to reverse the levered vs. unlevered ratio to make sense
            ])
        res = prob.solve(solver=cp.GLPK_MI)
        print("Spread: " + str(res))
        def idx(matrix): return int(next(filter(lambda v: v[1] > 0, enumerate(matrix.value)))[0])

        unlevered_contract = self.unlevered_smile[idx(unlevered_leg)]
        levered_contract = self.levered_smile[idx(levered_leg)]

        print(unlevered_contract.contract)
        print(unlevered_contract.price)

        print(levered_contract.contract)
        print(levered_contract.price)


        #bag = Bag(currency="USD", exchange="SMART", symbol="SPX", comboLegs=[
        #    ComboLeg(conId=short_contract.contract.conId, ratio=2, action='SELL', exchange='SMART'),
        #    ComboLeg(conId=upper_contract.contract.conId, ratio=1, action='BUY', exchange='SMART'),
        #    ComboLeg(conId=lower_contract.contract.conId, ratio=1, action='BUY', exchange='SMART')])

        return None


if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    futures = [load_scanner()]
    #futures = [run_scanner()]
    loop.run_until_complete(asyncio.wait(futures))
    loop.close()