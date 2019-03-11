import collections
from random import shuffle
import os
from os import listdir
from os.path import join

import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops            
import nvidia.dali.types as types


def is_jpeg(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg"])


def get_subdirs(directory):
    subdirs = sorted([join(directory,name) for name in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, name))])
    return subdirs


flatten = lambda l: [item for sublist in l for item in sublist]


class ExternalInputIterator(object):
    
    def __init__(self, imageset_dir, batch_size, random_shuffle=False):
        self.images_dir = imageset_dir
        self.batch_size = batch_size

        # First, figure out what are the inputs and what are the targets in your directory structure:
        # Get a list of filenames for the target (frontal) images
        self.frontals = np.array([join(imageset_dir, frontal_file) for frontal_file in sorted(os.listdir(imageset_dir)) if is_jpeg(frontal_file)])
        
        # Get a list of lists of filenames for the input (profile) images for each person
        profile_files = [[join(person_dir, profile_file) for profile_file in sorted(os.listdir(person_dir)) if is_jpeg(profile_file)] for person_dir in get_subdirs(imageset_dir)]
        
        # Build a flat list of frontal indices, corresponding to the *flattened* profile_files
        # The reason we are doing it this way is that we need to keep track of the multiple inputs corresponding to each target
        frontal_ind = []
        for ind, profiles in enumerate(profile_files):
            frontal_ind += [ind]*len(profiles)
        self.frontal_indices = np.array(frontal_ind)
        
        # Now that we have built frontal_indices, we can flatten profile_files
        self.profiles = np.array(flatten(profile_files))

        # Shuffle the (input, target) pairs if necessary: in practice, it is profiles and frontal_indices that get shuffled
        if random_shuffle:
            ind = np.array(range(len(self.frontal_indices)))
            shuffle(ind)
            self.profiles = self.profiles[ind]
            self.frontal_indices = self.frontal_indices[ind]

                
    def __iter__(self):
        self.i = 0
        self.n = len(self.frontal_indices)
        return self

    
    # Return a batch of (input, target) pairs
    def __next__(self):
        profiles = []
        frontals = []
        
        for _ in range(self.batch_size):
            profile_filename = self.profiles[self.i]
            frontal_filename = self.frontals[self.frontal_indices[self.i]]

            profile = open(profile_filename, 'rb')
            frontal = open(frontal_filename, 'rb')

            profiles.append(np.frombuffer(profile.read(), dtype = np.uint8))
            frontals.append(np.frombuffer(frontal.read(), dtype = np.uint8))

            profile.close()
            frontal.close()

            self.i = (self.i + 1) % self.n
        return (profiles, frontals)

    next = __next__


class ImagePipeline(Pipeline):
    '''
    Constructor arguments:  
    - imageset_dir: directory containing the dataset
    - image_size = 128: length of the square that the images will be resized to
    - random_shuffle = False
    - batch_size = 64
    - num_threads = 2
    - device_id = 0
    '''
    
    def __init__(self, imageset_dir, image_size=128, random_shuffle=False, batch_size=64, num_threads=2, device_id=0):
        super(ImagePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)  
        eii = ExternalInputIterator(imageset_dir, batch_size, random_shuffle)
        self.iterator = iter(eii)
        self.num_inputs = len(eii.frontal_indices)

        # The source for the inputs and targets
        self.input = ops.ExternalSource()
        self.target = ops.ExternalSource()
        
        # nvJPEGDecoder below accepts  CPU inputs, but returns GPU outputs (hence device = "mixed")
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        
        # The rest of pre-processing is done on the GPU
        self.res = ops.Resize(device="gpu", resize_x=image_size, resize_y=image_size)
        self.norm = ops.NormalizePermute(device="gpu", output_dtype=types.FLOAT,
                                         mean=[128., 128., 128.], std=[128., 128., 128.],
                                         height=image_size, width=image_size)
    
    
    # epoch_size = number of (profile, frontal) image pairs in the dataset
    def epoch_size(self, name = None):
        return self.num_inputs

    
    # Define the flow of the data loading and pre-processing
    def define_graph(self):    
        self.profiles = self.input(name="inputs")
        self.frontals = self.target(name="targets")
        profile_images = self.decode(self.profiles)
        profile_images = self.res(profile_images)
        profile_output = self.norm(profile_images)
        frontal_images = self.decode(self.frontals)
        frontal_images = self.res(frontal_images)
        frontal_output = self.norm(frontal_images)
        return (profile_output, frontal_output)

    
    def iter_setup(self):
        (images, targets) = self.iterator.next()
        self.feed_input(self.profiles, images)
        self.feed_input(self.frontals, targets)

