import argparse
import os
import cv2
import h5py
import numpy as np


def pack_h5(stack_test_dir, out_dir):
    #
    # Read one test stack image to determine shape of stack images in test set
    folders_test = os.listdir(stack_test_dir)

    test_sample_names = [os.path.join(stack_test_dir, folders_test[0], filename)
                         for filename in os.listdir(os.path.join(stack_test_dir, folders_test[0]))]
    test_stack_shape = (len(folders_test), len(test_sample_names)) + cv2.cvtColor(cv2.imread(test_sample_names[0]),
                                                                                  cv2.COLOR_BGR2RGB).astype(float).shape

    hdf5 = h5py.File(out_dir, mode='w')
    hdf5.create_dataset("stack_test", test_stack_shape, np.float32)
    print("Packing test set")
    # iterate over focus stacks
    for i, folder in enumerate(folders_test):
        # Determine stack image names
        stack_sample_names = [os.path.join(stack_test_dir, folder, filename)
                              for filename in os.listdir(os.path.join(stack_test_dir, folder))]
        # Read stack images
        stack_samples = np.asarray(
            [cv2.cvtColor(cv2.imread(stack_sample), cv2.COLOR_BGR2RGB).astype(float) for stack_sample in
             stack_sample_names])
        # Add to dataset
        hdf5["stack_test"][i] = stack_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stacktest", help="input directory containing testing focal stacks", type=str)
    parser.add_argument("outfile", help="h5 file to be written", type=str)
    args = parser.parse_args()

    pack_h5(args.stacktest, args.outfile)
