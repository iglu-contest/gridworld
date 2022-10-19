import numpy as np
from gridworld.data import IGLUDataset
from gridworld.tasks import Tasks

from tqdm import tqdm

dataset_rc2 = IGLUDataset(dataset_version="v0.1.0-rc2", force_parsing=False)
dataset_rc3 = IGLUDataset(dataset_version="v0.1.0-rc3", force_parsing=True)

print("Test: Verifying data in v0.1.0-rc2 and v0.1.0-rc3 is the same.")
task_ids = list(dataset_rc3.tasks.keys())
for key in tqdm(task_ids, desc="Structures", leave=False):
    assert len(dataset_rc2.tasks[key]) == len(dataset_rc3.tasks[key])

    for j in tqdm(range(len(dataset_rc3.tasks[key])), leave=False, desc="Tasks"):
        assert len(dataset_rc2.tasks[key][j]) >= len(dataset_rc3.tasks[key][j])

        tasks_new = list(dataset_rc3.tasks[key][j])
        tasks_orig = list(dataset_rc2.tasks[key][j])
        for k in range(len(tasks_new)):
            assert len(tasks_new[k]) == len(tasks_orig[k])
            assert tasks_orig[k].last_instruction == tasks_new[k].last_instruction
            assert tasks_orig[k].starting_grid == tasks_new[k].starting_grid
            assert np.all(tasks_orig[k].target_grid == tasks_new[k].target_grid)

print("Success!")

print("Test: Replaying block changes to build the target structure.")
task_ids = list(dataset_rc3.tasks.keys())
for key in tqdm(task_ids, desc="Structures"):
    for j in tqdm(range(len(dataset_rc3.tasks[key])), leave=False, desc="Tasks"):
        tasks = list(dataset_rc3.tasks[key][j])
        for k in range(len(tasks)):
            task = tasks[k]
            grid = {tuple((x,y,z)): bid for x,y,z,bid in task.starting_grid}

            for x, y, z, bid in task.block_changes:
                assert grid.get(tuple((x, y, z)), 0) != bid, "Already set! Redundant block changes?"
                grid[tuple((x, y, z))] = bid

            sorted(grid) == sorted(Tasks.to_sparse(task.target_grid.transpose((1,0,2))))

print("Success!")
