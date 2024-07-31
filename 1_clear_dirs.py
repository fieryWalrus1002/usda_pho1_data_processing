# This code snippet is designed to clear out existing data and plots in specific directories.
import os
import sys


def clear_dirs(
    base_dir, sub_dirs=["output_data", "output_plots", "output_protocol", "scripts"]
):
    print(f"Clearing {sub_dirs} in {base_dir}...")

    # join the base path and the sub_dirs?
    for d in sub_dirs:
        full_path = os.path.join(base_dir, d)
        if os.path.exists(full_path):
            for f in os.listdir(full_path):
                os.remove(os.path.join(full_path, f))
        else:
            os.makedirs(full_path)


if __name__ == "__main__":
    # if there are no arguments, use the current directory as base_dir
    if len(sys.argv) <= 1:
        base_dir = os.getcwd()
        clear_dirs(base_dir)
    elif len(sys.argv) == 2:
        base_dir = sys.argv[1]
        clear_dirs(base_dir)
    else:
        base_dir = sys.argv[1]
        sub_dirs = sys.argv[2:]
        clear_dirs(base_dir, sub_dirs)
