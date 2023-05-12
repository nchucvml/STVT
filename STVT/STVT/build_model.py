#from . import models
from STVT.models.STVT import STVT

def build_model(args):
    model = STVT(dataset=args.dataset)
    return model
