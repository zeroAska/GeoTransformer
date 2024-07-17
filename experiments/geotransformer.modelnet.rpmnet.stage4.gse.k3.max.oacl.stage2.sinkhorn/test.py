import os.path as osp
import time

from geotransformer.engine import SingleTester
from geotransformer.utils.common import get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        message = get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


def main_different_outlier():

    for angle in [ 45.0]:
        for noise in [0.0, 0.1]:
            for outlier in [0.0, 0.2]:
                for crop in [0.0, 0.05, 0.1, 0.2]:
                    #print("GeoTransformer ModelNet: angle {}, noise {}, outlier {}".format(angle, noise, outlier))
                    cfg = make_cfg()
                    cfg.data.rotation_magnitude = angle
                    cfg.test.noise_magnitude = noise
                    cfg.data.outlier_augmentation = outlier
                    cfg.data.keep_ratio = 1-crop
                    tester = Tester(cfg)
                    tester.run()
                    print("Last result: GeoTransformer ModelNet: angle {}, noise {}, outlier {}, cropping {}".format(angle, noise, outlier, crop))
                    print("====================")


if __name__ == '__main__':
    main() #_different_outlier()
