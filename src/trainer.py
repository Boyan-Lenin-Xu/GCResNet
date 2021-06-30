import os
import math
from decimal import Decimal
from xml.dom.minidom import parse
import utility
from torch.autograd import Variable
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
from PIL import Image

class SRTrainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


class DeblurTrainer():
    def __init__(self, options, loader, my_model, my_loss, ckp):
        self.options = options
        self.model = my_model
        self.loss = my_loss
        self.ckp = ckp
        self.mode = options.getElementsByTagName('mode')[0].childNodes[0].nodeValue
        if self.mode == 'Train':
            self.loader_train = loader.loader_train
        elif self.mode == 'Test':
            self.loader_test = loader.loader_test
        self.optimizer = utility.deblur_optimizer(options, self.model)
        self.test_only = options.getElementsByTagName('test_only')[0].childNodes[0].nodeValue

        if self.options.getElementsByTagName('retrain')[0].childNodes[0].nodeValue != 'False':
            self.optimizer.load()
        
        self.base_lr = float(options.getElementsByTagName('base_lr')[0].childNodes[0].nodeValue)
        self.print_every = 5
        self.epochs = int(options.getElementsByTagName('epoch_num')[0].childNodes[0].nodeValue)
        self.batch_size = int(options.getElementsByTagName('batch_size')[0].childNodes[0].nodeValue)
        self.save_results = options.getElementsByTagName('save_results')[0].childNodes[0].nodeValue
        self.rgb_range = 255

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (blur, sharp, _,) in enumerate(self.loader_train):
            blur, sharp = self.prepare(blur, sharp)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            cleaned = self.model(blur)
            loss = self.loss(cleaned, sharp)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.ckp.save(self, epoch, is_best=False)
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        print(len(self.loader_test))
        self.ckp.write_log('\nEvaluation:')
        #self.ckp.add_log(
        #    torch.zeros(1, len(self.loader_test), 1)
        #)
        self.model.eval()

        timer_test = utility.timer()
        #if self.save_results == 'True': self.ckp.begin_background()
        #print(self.loader_test[0])
        for idx_data, (blur, _, filename) in enumerate(self.loader_test):
            print(blur.size())
            lr = self.prepare(blur)
            print(lr[0].size())
            cleaned = self.model(lr[0])
            
            cleaned = utility.quantize(cleaned, self.rgb_range)

            #save_list = [cleaned]

            #if self.save_results == 'True':
            #    self.ckp.save_results(GoPro_test, filename[0], save_list, scale=1)

            #self.ckp.log[-1, idx_data, idx_scale] /= 1

            cleaned = cleaned.data.cpu().numpy()[0]
            #　res[res>1] = 1
            #　res[res<0] = 0
            #　res*= 255
            cleaned = cleaned.astype(np.uint8)
            cleaned = cleaned.transpose((1,2,0)) # BGR RGB
            #print(filename[0])
            Image.fromarray(cleaned).save(filename[0])
            print('finished test：' + filename[0]+ '    cost')



        #self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        #self.ckp.write_log('Saving...')

        #if self.save_results == 'True':
        #    self.ckp.end_background()

        #self.ckp.write_log(
        #    'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        #)

        #torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        return [a.to(device) for a in args]

    def terminate(self):
        if self.test_only == 'True':
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.epochs
