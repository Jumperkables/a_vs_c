__author__ = "Jumperkables"

# standard imports
import os

# third party imports

# local imports
import utils

def main():
    norm_dict_path = os.path.join("/home/jumperkables/a_vs_c", "misc", "all_norms.pickle")
    norm_dict = utils.load_pickle(norm_dict_path)
    breakpoint()


if __name__ == "__main__":
    main()
