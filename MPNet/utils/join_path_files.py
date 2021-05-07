import numpy
import os
import argparse

import numpy as np


def main(args):
    subfolders_list = os.listdir(args.folder)
    subfolders_list = [subfolder for subfolder in subfolders_list if not os.path.isfile(f"{args.folder}/{subfolder}")]
    
    join_context = args.context
    if join_context == "subfolder":
        for subfolder in subfolders_list:
            joined_file = []
            files_list = os.listdir(os.path.join(args.folder, subfolder))
            for file in files_list:
                file = os.path.join(args.folder, subfolder, file)
                joined_file.append(np.fromfile(file, dtype=float))
            
            joined_file = np.array(joined_file, dtype=object)
            save_stacked_array(f'{args.output}/{subfolder}', joined_file)
            # with open(os.path.join(args.output, f'{subfolder}.dat'), 'wb') as f:
            #     np.save(f, joined_file)
    
    elif join_context == "all":
        paths_list = [os.listdir(os.path.join(args.folder, subfolder)) for subfolder in subfolders_list]
        files_list = [os.path.join(args.folder, subfolder, path) for subfolder, arr in zip(subfolders_list, paths_list)
                      for path in arr]
        joined_file = []
        for file in files_list:
            joined_file.append(np.fromfile(file, dtype=float))
        joined_file = np.array(joined_file, dtype=object)
        
        envs_mapping = np.array([os.path.dirname(f) for f in files_list])
        
        save_stacked_array(f'{args.output}', joined_file, envs_mapping)


def stack_ragged(array_list, axis=0):
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx


def save_stacked_array(fname, array_list, envs_mapping, axis=0):
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(fname, array=stacked, index=idx, envs=envs_mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="", type=str)
    parser.add_argument("--context", default="subfolder", type=str)
    parser.add_argument("--output", default="", type=str)
    
    args = parser.parse_args()
    
    main(args)
