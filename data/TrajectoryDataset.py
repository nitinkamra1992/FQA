import os
import errno
import pickle

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd


class TrajectoryDataset(data.Dataset):
    def __init__(self, data_files,
                 max_length=20,
                 min_length=10,
                 to_tensor=True,
                 random_rotation=False,
                 burn_in_steps=8,
                 skip_frames=0,
                 frame_shift=1,
                 no_filter=False,
                 use_pickled=False):

        # Max and min length of consecutive sequence
        self.max_length = max_length
        self.min_length = min_length
        self.burn_in_steps = burn_in_steps
        self.skip_frames = skip_frames
        self.frame_shift = frame_shift
        self.no_filter = no_filter

        # If true return torch.Tensor, or otherwise return np.array
        self.to_tensor = to_tensor

        # If true, apply random rotation (around origin) to data points
        self.random_rotation = random_rotation

        # List containing raw data for each scene
        self.scenes = {}

        # List of dictionary containing:
        #   "scene_id": (int)
        #   "start": (int)
        #   "interval": (int)
        self.frame_seqs = []

        # Load data from files
        self.load_data(data_files, use_pickled)


    def load_data(self, data_files, use_pickled=False):

        # Each data file contains raw data from a single scene
        for scene_file in data_files:
            pickle_path = self.get_pickle_path(scene_file)

            if use_pickled and os.path.exists(pickle_path):
                scene = _load_data(scene_file)
                frame_seqs = _load_pickle(pickle_path)
            else:
                scene, frame_seqs = self.load_from_file(scene_file)
                _dump_pickle(frame_seqs, pickle_path)

            self.frame_seqs += frame_seqs
            self.scenes[scene_file] = scene


    def get_pickle_path(self, scene_file):
        """ Return pickle path for the scene_file """
        dirname = os.path.dirname(scene_file)
        filename = os.path.splitext(os.path.basename(scene_file))[0]

        tokens = [filename, str(self.max_length), str(self.min_length), str(self.skip_frames)]
        if self.burn_in_steps > 0:
            tokens.append(str(self.burn_in_steps))

        if self.frame_shift > 1:
            tokens.append('fs'+str(self.frame_shift))

        pickle_name = '-'.join(tokens) + '.pkl'
        pickle_path = os.path.join(dirname, 'pickles', pickle_name)

        return pickle_path


    def load_from_file(self, scene_file):
        """ Processes and load data from scene_file
        Args:
            scene_file - Path to the file to be loaded
        Returns:
            scene - Pandas DataFrame containing data from scene_file
            frame_seqs - List containing info about frame sequences
        """
        scene = _load_data(scene_file)
        interval = _get_frame_interval(scene)
        frame_ids = _get_frame_ids(scene)
        frame_seqs = []

        if self.skip_frames > 0:
            interval = self.skip_frames

        # For each frame in a scene,
        # check if there are at least min_length frames
        # starting from the current frame.
        for frame in frame_ids[::self.frame_shift]:
            start = int(frame)
            end = start + (self.max_length * interval)

            # get data sequence between the "start" and "end" frames (inclusive)
            data_seq = _get_data_sequence_between(scene, start, end, interval)
            num_frames = len(_get_frame_ids(data_seq))

            # Check if there are at least min_length frames
            if num_frames >= self.min_length + 1: # 1 extra frame checked for to incorporate both data, preds
                # add the info about frame sequence to frame_seqs
                frame_seqs.append({
                        "scene_id": scene_file,
                        "start": start,
                        "interval": interval
                    })

        return scene, frame_seqs

    def __getitem__(self, index):
        """
        N : number of agents
        L : max sequence length
        D : dimension of spatial trajectory coordinate
        Return:
            source - tensor of shape (N, L, D)
            target - tensor of shape (N, L, D)
            mask - tensor of shape (N, L)
        """
        # Get frame sequence and scene
        frame_seq = self.frame_seqs[index]
        scene = self.scenes[frame_seq['scene_id']]
        interval = frame_seq['interval']

        start = frame_seq['start'] 
        end = start + (self.max_length * interval)

        # Get data sequence of length (max_length + 1) from the scene
        data_seq = _get_data_sequence_between(scene, start, end, interval)

        # Convert raw data sequence to arrays
        data, mask = _to_array(data_seq, self.max_length + 1, start, interval)

        # Filtering on agents
        if (not self.no_filter):
            m = mask[:,:,0]
            # Remove agents present for less than half the trajectory
            m[np.sum(m, axis=1) / m.shape[1] < 1/2] = 0

            # Remove agents not present at time == br - 3
            if self.burn_in_steps >= 3:
                m[m[:,self.burn_in_steps-3] == 0] = 0
            mask = np.tile(np.expand_dims(m,2), (1,1,2))

        # Get source and target data and masks
        source = data[:, 0:self.max_length, :]
        target = data[:, 1:self.max_length+1, :]
        source_mask = mask[:, 0:self.max_length, :]
        target_mask = mask[:, 1:self.max_length+1, :]

        if (self.random_rotation):
            theta = 2 * np.pi * np.random.random()
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            source = np.matmul(source, R)
            target = np.matmul(target, R)

        if (self.to_tensor):
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            source_mask = torch.Tensor(source_mask)
            target_mask = torch.Tensor(target_mask)

        return source, target, source_mask, target_mask

    def __len__(self):
        return len(self.frame_seqs)


# Helper functions
def _load_data(data_file):
    """ Read data from a file and returns a pd.DataFrame """
    df = pd.read_csv(data_file, sep="\s+", header=None)
    df.columns = ["frame_id", "agent_id", "x", "y"]
    return df

def _load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def _dump_pickle(data, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'wb') as f:
        pickle.dump(data, f)

#def _get_data_sequence_between(df, start, end):
#    """ Returns all data with frame_id between start and end (inclusive) """
#    return df.loc[df.frame_id.between(start, end)]

def _get_data_sequence_between(df, start, end, interval):
    """ Returns all data with frame_id between start and end (inclusive) """
    data_seq = df.loc[df.frame_id.between(start, end)]
    data_seq = data_seq[(data_seq.frame_id - start) % interval == 0]
    return data_seq 

def _get_frame_interval(df):
    """ Calculate frame interval of the DataFrame df.
        Assumes that the first two frame_ids are consecutive
    """
    frame_ids = sorted(list(_get_frame_ids(df)))
    return int(frame_ids[1] - frame_ids[0])

def _get_frame_ids(df):
    """ Returns unique frame_ids in the DataFrame df """
    return df.frame_id.unique()

def _get_agent_ids(df):
    """ Returns unique agent_ids in df """
    return df.agent_id.unique()

def _to_array(df, max_length, start, interval, dim=2):
    """ Convert input DataFrame df to 3-dimensional Numpy array"""
    agents = sorted(list(_get_agent_ids(df)))
    num_agents = len(agents)

    # First create an array of shape (N, L, D) filled with inf.
    # We will replace infs with 0 after we compute mask
    array = np.full((num_agents, max_length, dim), np.inf)

    # Compute indexs to fill out
    agent_idxs = (df.agent_id.apply(agents.index)).astype(int)
    frame_idxs = ((df.frame_id - start) / interval).astype(int)
    coords = df[['x', 'y']]
    assert len(coords) == len(frame_idxs) and len(coords) == len(agent_idxs)

    # Fill out arrays with coordinates
    array[agent_idxs, frame_idxs] = coords

    # Compute mask where mask[i, j] == 0 for array[i, j] == inf
    mask = (array != np.inf).astype(int)

    # Finally replace inf with 0
    array[array == np.inf] = 0

    return array, mask

