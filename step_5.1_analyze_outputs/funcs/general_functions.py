import pandas as pd
import numpy as np
import pickle as pkl

class gen_funcs:
    def get_pixels(self,pixels_fname):
        '''
        Function that gets the Catchment-CN4.5 pixels that we are currently
        working with.
        Inputs:
            pixels_fname: fname of the step_1 file where the pixels that we
            need are saved.
        Ouputs:
            pixels: A numpy array containing the pixel numbers as ints
        '''
        # get the pixels that we are going to run as defined in step 1
        pixels = pd.read_csv(pixels_fname,header=None)
        # turn this into an np array
        pixels = np.array(pixels).astype(int)
        # make a nice np array
        pixels = pixels.T
        pixels = pixels[0]
        # return the pixels
        return pixels
    def get_intersection_info(self,fname):
        with open(fname,'rb') as f:
            out = pkl.load(f)
        return out
