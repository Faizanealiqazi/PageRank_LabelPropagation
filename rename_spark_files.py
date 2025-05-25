import glob
import shutil
import os
def rename_single_csv(dir_path, final_name):
    part_file = glob.glob(f"{dir_path}/part-*.csv")[0]
    shutil.move(part_file, f"{dir_path}/{final_name}.csv")

    # Optionally delete _SUCCESS file
    success_file = f"{dir_path}/_SUCCESS"
    if os.path.exists(success_file):
        os.remove(success_file)

