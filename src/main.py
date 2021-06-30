import torch
import xml.dom.minidom
from xml.dom.minidom import parse
import utility
import data
import model
import loss
from option import args
from trainer import SRTrainer, DeblurTrainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    #global model
    option_file = parse("./options.xml")
    options = option_file.getElementsByTagName("options")[0]

    if args.mode == 'deblur':
        options = options.getElementsByTagName('Deblurring')[0]
        #print(options)
        #This is the typical method to use XML format
        if options.getElementsByTagName('mode')[0].childNodes[0].nodeValue == 'Train':
            loader = data.DeblurData(options)
            _model = model.DeblurModel(options, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            Trainer = DeblurTrainer(options, loader, _model, _loss, checkpoint)
            while not Trainer.terminate():
                Trainer.train()
            checkpoint.done()
        else:
            loader = data.DeblurData(options)
            _model = model.DeblurModel(options, checkpoint)
            _loss = 0
            Trainer = DeblurTrainer(options, loader, _model, _loss, checkpoint)
            while not Trainer.terminate():
                Trainer.test()
            checkpoint.done()

    if args.mode == 'super-resolution':
        if args.data_test == ['video']:
            from videotester import VideoTester
            _model = model.SRModel(args, checkpoint)
            t = VideoTester(args, _model, checkpoint)
            t.test()
        else:
            if checkpoint.ok:
                loader = data.Data(args)
                _model = model.SRModel(args, checkpoint)
                _loss = loss.Loss(args, checkpoint) if not args.test_only else None
                Trainer = SRTrainer(args, loader, _model, _loss, checkpoint)
                while not Trainer.terminate():
                    Trainer.train()
                    Trainer.test()

                checkpoint.done()

if __name__ == '__main__':
    main()

