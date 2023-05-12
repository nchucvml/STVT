import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
import shutil

def adjust_learning_rate(args, optimizer, epoch, batch_idx, data_nums, type):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / data_nums
        lr_adj = 1. * (epoch / args.warmup_epochs)
    elif type == "linear":
        if epoch < 30 + args.warmup_epochs:
            lr_adj = 1.
        elif epoch < 60 + args.warmup_epochs:
            lr_adj = 1e-1
        elif epoch < 90 + args.warmup_epochs:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
    elif type == "cosine":
        run_epochs = epoch - args.warmup_epochs
        total_epochs = args.epochs - args.warmup_epochs
        T_cur = float(run_epochs * data_nums) + batch_idx
        T_total = float(total_epochs * data_nums)

        lr_adj = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj
def label_smoothing(pred, target, eta=0.1):
    '''
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    '''
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)

def save_model(model, args, best_fscore_k, epoch):
    if os.path.exists("./STVT/model/"+args.dataset+"/model_" + str(args.roundtimes) + "_roundtimes"):
        shutil.rmtree("./STVT/model/"+args.dataset+"/model_" + str(args.roundtimes) + "_roundtimes")
    if not os.path.exists("./STVT/model/"+args.dataset+"/model_" + str(args.roundtimes)+"_roundtimes"):
        os.mkdir("./STVT/model/"+args.dataset+"/model_" + str(args.roundtimes)+"_roundtimes")
    path = "./STVT/model/"+args.dataset+"/model_"+str(args.roundtimes)+"_roundtimes/"+str(args.dataset)+"_"+str(epoch)+"_"+str(best_fscore_k)+".pth"
    torch.save(model.state_dict(), path)
def dist_save_model(model, optimizer, epoch, ngpus_per_node, args):
    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        os.system('mkdir -p {}'.format(args.work_dirs))
        if optimizer is not None:
            torch.save(
                {
                    'net': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'epoch': epoch,
                },
                os.path.join(args.work_dirs, '{}.pth'.format(epoch)),
            )
        else:
            torch.save(
                {'net': model.state_dict(), 'epoch': epoch},
                os.path.join(args.work_dirs, '{}.pth'.format(epoch)),
            )


def load_model(network, args):
    if not os.path.exists(args.work_dirs):
        print("No such working directory!")
        raise AssertionError

    pths = [
        pth.split('.')[0] for pth in os.listdir(args.work_dirs) if 'pth' in pth
    ]
    if len(pths) == 0:
        print("No model to load!")
        raise AssertionError

    pths = [int(pth) for pth in pths]
    if args.test_model == -1:
        pth = -1
        if pth in pths:
            pass
        else:
            pth = max(pths)
    else:
        pth = args.test_model
    try:
        if args.distributed:
            loc = 'cuda:{}'.format(args.gpu)
            model = torch.load(
                os.path.join(args.work_dirs, '{}.pth'.format(pth)),
                map_location=loc,
            )
    except:
        model = torch.load(os.path.join(args.work_dirs, '{}.pth'.format(pth)))
    try:
        network.load_state_dict(model['net'], strict=True)
    except:
        network.load_state_dict(convert_model(model['net']), strict=True)
    return True


def resume_model(network, optimizer, args):
    print("Loading the model...")
    path = "C:/Users/forUniversity-2080-3/ViT-main/model"
    #if not os.path.exists(args.work_dirs):
    if not os.path.exists(path):
        print("No such working directory!")
        return 0
    pths = [
        #pth.split('.')[0] for pth in os.listdir(args.work_dirs) if 'pth' in pth
        pth.split('.')[0] for pth in os.listdir(path) if 'pth' in pth
    ]
    if len(pths) == 0:
        print("No model to load!")
        return 0
    pths = [int(pth) for pth in pths]
    if args.test_model == -1:
        pth = max(pths)
    else:
        pth = args.test_model
    try:
        if args.distributed:
            loc = 'cuda:{}'.format(args.gpu)
            model = torch.load(
                #os.path.join(args.work_dirs, '{}.pth'.format(pth)),
                os.path.join(path, '{}.pth'.format(pth)),
                map_location=loc,
            )
    except:
        #model = torch.load(os.path.join(args.work_dirs, '{}.pth'.format(pth)))
        model = torch.load(os.path.join(path, '{}.pth'.format(pth)))
    try:
        network.load_state_dict(model['net'], strict=True)
    except:
        network.load_state_dict(convert_model(model['net']), strict=True)
    optimizer.load_state_dict(model['optim'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                try:
                    state[k] = v.cuda(args.gpu)
                except:
                    state[k] = v.cuda()
    return model['epoch']


def convert_model(model):
    new_model = {}
    for k in model.keys():
        new_model[k[7:]] = model[k]
    return new_model


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s
