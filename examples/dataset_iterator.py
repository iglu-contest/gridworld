from gridworld.data import CDMDataset, IGLUDataset
import gridworld
from time import sleep
import sys
sys.path.insert(0, '../')

DATASET_VERSION = "v0.1.0-rc1"

# cdm_dataset = CDMDataset(task_kwargs={'invariant': False})
iglu_dataset = IGLUDataset(
    dataset_version=DATASET_VERSION, task_kwargs={'invariant': False})

for task_id, n, m, subtask in iglu_dataset:
    print(f'structure id: {task_id}, session id: {n}, '
          f'substructure id: {m}, instruction len: {len(subtask.last_instruction)}')
