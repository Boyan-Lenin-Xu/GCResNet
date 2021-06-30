from importlib import import_module
#from dataloader import MSDataLoader
import xml.dom.minidom
from xml.dom.minidom import parse
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )

class DeblurData:
    def __init__(self, options):
        if options.getElementsByTagName('mode')[0].childNodes[0].nodeValue == 'Train':
            dataset_name = options.getElementsByTagName('dataset_name')[0].childNodes[0].nodeValue
            self.batch_size = int(options.getElementsByTagName('batch_size')[0].childNodes[0].nodeValue)
            m = import_module('data.' + dataset_name)
            self.dataset = getattr(m, dataset_name)(options, name='GoPro', train=True)
            self.loader_train = dataloader.DataLoader(self.dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True, num_workers=8,
                                                      )
        #No test when training deblur models
        if options.getElementsByTagName('mode')[0].childNodes[0].nodeValue == 'Test':
            dataset_name = options.getElementsByTagName('dataset_name')[0].childNodes[0].nodeValue
            self.batch_size = 1
            m = import_module('data.' + dataset_name)
            self.dataset = getattr(m, dataset_name)(options, name='GoPro', train=False)
            self.loader_test = dataloader.DataLoader(self.dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False, num_workers=8,
                                                      )
