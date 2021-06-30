import os
import glob
from data import deblurdata

class GoPro(deblurdata.DBData):
    def __init__(self, options, name='GoPro', train=True):
        super(GoPro, self).__init__(
            options, name=name, train=train, benchmark=False
        )
        #self.mode = options.getElementsByTagName('mode')[0].childNodes[0].nodeValue

    def _scan(self):
        super(GoPro, self)._scan()
        if self.mode == 'Train':
            name_list = sorted(glob.glob(os.path.join(self.dir_data, 'GoPro', 'GOPR*')))
            self.dir_blur = []
            self.dir_sharp = []
            for name in name_list:
                in_images = os.listdir(os.path.join(name, 'blur'))
                for in_image in in_images:
                    dir_blur = os.path.join(name, 'blur', in_image)
                    self.dir_blur.append(dir_blur)
                    dir_sharp = os.path.join(name, 'sharp', in_image)
                    self.dir_sharp.append(dir_sharp)
        
            return self.dir_sharp, self.dir_blur

        elif self.mode == 'Test':
            name_list = sorted(glob.glob(os.path.join(self.dir_data, 'GoPro_test', 'GOPR*')))
            self.dir_blur = []
            for name in name_list:
                in_images = os.listdir(name)
                for in_image in in_images:
                    dir_blur = os.path.join(name, in_image)
                    self.dir_blur.append(dir_blur)

            return 0, self.dir_blur

    def _set_filesystem(self):
        super(GoPro, self)._set_filesystem()
        test_data = os.path.join(self.dir_data, 'GoPro_test')
        test_data_dir_list = os.listdir(test_data)
        for test_dir in test_data_dir_list:
            if not os.path.exists('./test/GoPro/' + test_dir):
                #os.system(r"touch {}".format('./test/GoPro/' + test_dir))
                os.makedirs('./test/GoPro/' + test_dir)

