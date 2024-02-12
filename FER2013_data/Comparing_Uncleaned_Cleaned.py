import os


def get_unique_files(dir1, dir2):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    unique_to_dir1 = files1 - files2

    return unique_to_dir1


def print_unique_files(unique_files, directory):
    print(f"Files unique to {directory}:")
    for file in unique_files:
        print(file)


def main():
    dir1 = input("Enter the first directory path: ")
    dir2 = input("Enter the second directory path: ")

    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print("One or both directories do not exist.")
        return

    print_unique_files(get_unique_files(dir1, dir2), dir1)



if __name__ == "__main__":
    main()
