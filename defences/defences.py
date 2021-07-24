from defences.pdgan import PDGAN

class DefServer:
    def __init__(self, hlpr, user_id=999, compromised=False, train_loader=None):
        self.hlpr = hlpr
        self.user_id = user_id
        self.compromised = compromised
        self.train_loader = train_loader
        self.defence_utility = None

    def set_defence(self, defence):
        if "PDGAN" in defence:
            self.defence_utility = PDGAN(hlpr=self.hlpr)
            return self.defence_utility.netD

    def defend(self, local_update_list, global_model, epoch):
        self.train_loader = self.hlpr.task.train_loader
        benign_update_list = self.defence_utility.run_defence(self.train_loader, global_model, local_update_list, epoch)
        return benign_update_list