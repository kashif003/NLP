# checking number of papers:
import os

def count_items_in_dir(directory_path):
    # Ensure the directory actually exists first
    if not os.path.exists(directory_path):
        print("Directory does not exist.")
        return

    file_count = 0
    dir_count = 0

    # os.listdir gives just the names, so we join them with the base path
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        
        if os.path.isfile(full_path):
            file_count += 1
        elif os.path.isdir(full_path):
            dir_count += 1

    print(f"In '{directory_path}':")
    print(f"📂 Total Directories: {dir_count}")
    print(f"📄 Total Files: {file_count}")

# Example Usage: Check the current folder
count_items_in_dir("./data/latex_source")
count_items_in_dir("./data/pdf_source")
