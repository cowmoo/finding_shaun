from ib_insync import *


class Connection:
    class __Singleton:
        def __init__(self):
            self.ib = IB()
            self.ib.client.MaxRequests = 40
            self.ib.connect("127.0.0.1", 4001, clientId=1)
            self.qualifiedContracts = {}

        def __str__(self):
            return repr(self) + self.val

        def __getattr__(self, name):
            return getattr(self.ib, name)

        def qualifyContract(self, contract):
            if contract.conId not in self.qualifiedContracts.keys():
                self.qualifiedContracts[contract.conId] = self.ib.qualifyContracts(*[contract])[0]

            return self.qualifiedContracts[contract.conId]

    instance = None

    def __init__(self):
        if not Connection.instance:
            Connection.instance = Connection.__Singleton()

    def __getattr__(self, name):
        return getattr(self.instance, name)