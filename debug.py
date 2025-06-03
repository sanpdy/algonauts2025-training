from __future__ import print_function
import h5py

def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * (tabs + tab_step) + '- Dataset:', v.name, ', Shape:', v.shape, ', dtype:', v.dtype)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)

scan_hdf5('/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features/whisper_w16.h5', recursive=True, tab_step=2)
