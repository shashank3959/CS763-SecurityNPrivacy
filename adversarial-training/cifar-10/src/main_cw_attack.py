import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision as tv

from time import time
from model import WideResNet
from attack import FastGradientSignUntargeted, triplet_loss, npairs_loss,carlini_wagner_L2
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model,flop_benchmark, cw_adv_accuracy
from utils.flop_benchmark import get_model_infos

from argument import parser, print_args

from collections import OrderedDict
import math

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[50, 100, 150], 
                                                         gamma=0.1)
        _iter = 0

        begin_time = time()

        # Loss mixer: ce_loss_orig + (lambda1 * ce_loss_adversarial) + (lambda2 * triplet_loss)
        lambda1 = 0.6
        lambda2 = 0.5 # 1 for triplet, 0.5 for npairs for now

        for epoch in range(1, args.max_epoch+1):
            #scheduler.step()
            for data, label in tr_loader:

                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # Outputs from original samples
                    output = model(data, _eval=False)

                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack(model,data, label,to_numpy=False)

                    # Outputs from adversarial samples
                    adv_output = model(adv_data, _eval=False)

                    # Shape of pred: batch_size
                    # Predicted labels on adversarial samples
                    pred = torch.max(adv_output, dim=1)[1]

                    # Shape of output: batch_size * (64*widen_factor)
                    X_output = model(data, _eval=False, _prefinal=True)
                    X_adv_output = model(adv_data, _eval=False, _prefinal=True)

                    #tloss = triplet_loss(X_output, X_adv_output, label, pred, logger)
                    tloss = npairs_loss(X_output, X_adv_output, label)
                else:
                    output = model(data, _eval=False)
                    tloss = 0

                ce_loss = F.cross_entropy(output, label)
                if adv_train:
                    # Adding loss due to adversarial samples in the mix
                    ce_loss = ce_loss + lambda1 * F.cross_entropy(adv_output, label)
                    ce_loss = ce_loss / (1 + lambda1)

                loss = ce_loss + lambda2 * tloss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()
                    logger.info('Total Loss logger: %.3f, CE Loss: %.3f, Triplet loss: %.3f' % (loss, ce_loss, tloss))

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred_stand = torch.max(stand_output, dim=1)[1]

                        std_acc = evaluate(pred_stand.cpu().numpy(), label.cpu().numpy()) * 100

                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        
                        adv_data = self.attack(model,data, label,to_numpy=False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()

                    print('%.3f' % (t2 - t1))

                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time()-begin_time, loss.item()))

                    logger.info('standard acc: %.3f %%, robustness acc: %.3f %%' % (
                        std_acc, adv_acc))

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                if _iter % args.n_store_image_step == 0:
                    tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0), 
                                        os.path.join(args.log_folder, 'images_%d.jpg' % _iter), 
                                        nrow=16)

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % _iter)
                    save_model(model, file_name)

                _iter += 1

            # Update the scheduler
            scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) \
                    +'='*20)
                logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    va_acc, va_adv_acc, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')


    def test(self, model, loader, adv_test=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack(model,data, label,to_numpy=False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num

def cw_attack_test(model,te_loader):
    model.eval()
    cw_acc = 0.0
    te_acc = 0.0
    test_num=0
    adv_num = 0
    total_te_acc = 0.0
    total_cw_acc=0.0
    mean=[0]
    std=[1]

    inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                  max((1 - m) / s for m, s in zip(mean, std)))
                  
    for data, label in te_loader:
       te_acc = 0.0
       data, label = tensor2cuda(data), tensor2cuda(label)       
       test_output = model(data, _eval=True)       
       test_pred = torch.max(test_output, dim=1)[1]   
       te_acc = evaluate(test_pred.cpu().numpy(), label.cpu().numpy(), 'sum')       
       total_te_acc += te_acc        
       test_num += test_output.shape[0]
       
       #######################################################################
       # Adversarial Samples       
       cw_acc = 0.0
       attack = carlini_wagner_L2.L2Adversary(targeted=False,
                                              confidence=0.0,
                                              search_steps=2,
                                              box=inputs_box,
                                              optimizer_lr=0.7)
       adv_data = attack(model,data, label,to_numpy=False)
       adv_output = model(adv_data, _eval=True)
       adv_num += adv_output.shape[0]
       adv_pred = torch.max(adv_output, dim=1)[1]
       cw_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
       total_cw_acc+=cw_acc
    
       print ("Total Test Samples",test_num)
       print ("Total Adversarial Samples",adv_num)
       print ("Accuracy before CW Attack",((total_te_acc / test_num)*100))
       print ("Accuracy after CW Attack",((total_cw_acc / adv_num)*100))
       exit()
    

def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    # Using a WideResNet model
    model = WideResNet(depth=34, num_classes=10, widen_factor=1, dropRate=0.0)
    flop, param = get_model_infos(model, (1, 3, 32, 32))
    logger.info('Model Info: FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
     
    mean=[0]
    std=[1]

    inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                  max((1 - m) / s for m, s in zip(mean, std)))

    attack = carlini_wagner_L2.L2Adversary(targeted=False,
                                               confidence=0.0,
                                               search_steps=2,
                                               box=inputs_box,
                                               optimizer_lr=0.7)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)
    
    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                    (4,4,4,4), mode='constant', value=0).squeeze()),
                tv.transforms.ToPILImage(),
                tv.transforms.RandomCrop(32),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=True, 
                                       transform=transform_train, 
                                       download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        pass
    elif args.todo =='cw_test':
        model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
        print (model)
        model.load_state_dict(torch.load(args.cw_attack_modelpath, map_location='cpu'))
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)        
        cw_attack_test(model,te_loader)
        print (model)   
    
    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
