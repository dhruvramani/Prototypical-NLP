import argparse
import numpy as np


class ArgParser():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-r", "--root", type=str, default="../", help="root directory")
        parser.add_argument("-ds", "--dataset", type=str, default="../dataset", help="directory with the dataset")

        parser.add_argument("-nep", "--epoch", type=int, default=100, help="no of epochs")
        parser.add_argument("-iter", "--iterations", type=int, default=100, help="no of iterations")
        parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")

        parser.add_argument("-nCTr", "--classes_per_iter", type=int, default=60, help="total no of classes - training")
        parser.add_argument("-nsTr", "--support_training_samples", type=int, default=5, help="no of samples to calculate prototype - training")
        parser.add_argument("-nqTr", "--query_training_examples", type=int, default=5, help="no of samples to optimize loss - training")

        parser.add_argument("-nCVl", "--classes_per_iter_val", type=int, default=5, help="total no of classes - validation")
        parser.add_argument("-nsVl", "--support_validation_samples", type=int, default=5, help="no of samples to calculate prototype - validation")
        parser.add_argument("-nqVl", "--query_validation_examples", type=int, default=5, help="no of samples to optimize loss - validation")

        parser.add_argument("--cuda", type=self.str_to_bool, default=False, help="cuda?")

        self.parser = parser

    def str_to_bool(self, string):
        return bool(string.lower() == "true")

    def get_parser(self):
        return self.parser

# Checking
if __name__ == '__main__':
    parse = ArgParser()
    print(parse.get_parser())