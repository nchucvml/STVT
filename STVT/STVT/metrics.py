'''
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n
'''

class Metric(object):
    """Computes and stores the average and current value"""

    def __init__(self,name):
        self.name = name
        self.reset()

    def reset(self):
        self.loss = 0
        #self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, loss):
        self.loss = loss
        self.sum += loss * 1
        self.count += 1
        #self.avg = self.sum / self.count

    @property
    def avg(self):
        # print("self.count: "+str(self.count))
        #print("self.sum: " + str(self.sum))
        return self.sum / self.count

