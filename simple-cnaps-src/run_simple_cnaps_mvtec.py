import torch
import numpy as np
import argparse
import os
import sys
from utils import print_and_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy
from simple_cnaps import SimpleCnaps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings

from mvtectools.mvtec_dataloader import MVTecDataLoader

NUM_TRAIN_TASKS = 20000
NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 1000
VALIDATION_FREQUENCY = 1000

def main():
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.imagenet_subset = MVTecDataLoader(self.args.dataset_dir, shot_num=self.args.shots, way_num=self.args.ways,
                                               episode_test_sample_num=self.args.testnum)

        self.imagenet_subset.generate_data_list(phase='test', episode_num=NUM_TEST_TASKS)
        self.imagenet_subset.load_list(phase='test')

        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = SimpleCnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args, mt=True).to(self.device)
        model.train()  # set encoder is always in train mode to process context data
        model.feature_extractor.eval()  # feature extractor is always in eval mode
        if use_two_gpus:
            model.distribute_model()
        return model

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--pretrained_resnet_path", default="../model-checkpoints/pretrained_resnets/pretrained_resnet_mini_tiered_with_extra_classes.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--shots", type=int, default=1,
                            help="Number of shots in the task.")
        parser.add_argument("--ways", type=int, default=5,
                            help="Number of ways in the task.")
        parser.add_argument("--testnum", type=int, default=10,
                            help="Number of test examples per class in the task.")
        parser.add_argument("--dataset-dir", help="Dataset directory")
                            
        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        validation_accuracy = 0
        with tf.compat.v1.Session(config=config) as session:
            self.test(self.args.test_model_path, session)

            self.logfile.close()

    def test(self, path, session):
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

        with torch.no_grad():
            accuracies = []
            for idx in range(NUM_TEST_TASKS):
                context_images, context_labels, target_images, target_labels = \
                    self.imagenet_subset.get_batch(phase='test', idx=idx)
                context_images = torch.from_numpy(context_images).permute(0,3,1,2).cuda(0).float()
                context_labels = torch.argmax(torch.from_numpy(context_labels), dim=1).cuda(0)
                target_images = torch.from_numpy(target_images).permute(0,3,1,2).cuda(0).float()
                target_labels = torch.argmax(torch.from_numpy(target_labels), dim=1).cuda(0)
                target_logits = self.model(context_images, context_labels, target_images)
                accuracy = self.accuracy_fn(target_logits, target_labels)
                accuracies.append(accuracy.item())
                del target_logits
                if idx + 1 % 100 == 0:
                    print(f"{idx + 1} tasks done")
            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            print("Test Accuracy:", accuracy, "Confidence:", confidence)

    def use_two_gpus(self):
        use_two_gpus = False
        if self.args.feature_adaptation == "film+ar":
            use_two_gpus = True  # film+ar model does not fit on one GPU, so use model parallelism
        return use_two_gpus

if __name__ == "__main__":
    main()
