#! /usr/bin/python3

import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.metricseval.DDFFEval as DDFFEval

if __name__ == "__main__":
    #Set parameters
    image_size = (383, 552)
    filename_testset = "ddff-dataset-test.h5"
    checkpoint_file = "ddffnet-cc3-snapshot-121256.npz"

    #Create validation reader
    tmp_datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=None, stack_key="stack_test", disp_key="disp_val")

    #Create PSPDDFF evaluator
    evaluator = DDFFEval.DDFFEval(checkpoint_file, focal_stack_size=tmp_datareader.get_stack_size())
    #Evaluate
    metrics = evaluator.evaluate(filename_testset, image_size=image_size)
    print(metrics)
