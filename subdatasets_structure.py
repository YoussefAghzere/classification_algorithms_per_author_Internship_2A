import glob
import os
import shutil

"""folder = '/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author'
authors_dirs = glob.glob(folder + "/*")
print(authors_dirs)


for dir in authors_dirs:
    basename = dir.split("/")[-1]
    os.makedirs(f"{dir}/{basename}_Reedsy_Prompts_short_stories")
    os.makedirs(f"{dir}/{basename}_Reedsy_Prompts_short_stories_AI_Human_Style")
    os.makedirs(f"{dir}/{basename}_Reedsy_Prompts_short_stories_AI_Person_style")
    os.makedirs(f"{dir}/{basename}_Reedsy_Prompts_short_stories_AI_version")
    os.makedirs(f"{dir}/{basename}_Stories_AI")
    os.makedirs(f"{dir}/{basename}_Stories_AI_human_style")
    os.makedirs(f"{dir}/{basename}_Stories_AI_person_style")

"""
"""datasets_folders = ['/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Reedsy_Prompts_short_stories',
                    '/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Reedsy_Prompts_short_stories_AI_Human_Style',
                    '/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Reedsy_Prompts_short_stories_AI_Person_style',
                    '/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Reedsy_Prompts_short_stories_AI_version',
                    '/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Stories_AI_person_style']
folder = datasets_folders[4]
authors_folders = glob.glob(folder + "/*")
for f in authors_folders:
        basename = f.split("/")[-1]
        txt_files = glob.glob(f + "/*.txt")
        for file in txt_files:
            shutil.copy(file, f"/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author/{basename}/{basename}_Stories_AI_person_style/{file.split('/')[-1]}")

"""
datasets_folders = ['/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Stories_AI',
                    '/home/youssef/Desktop/classification_algorithms_per_author/DATASETS/Stories_AI_human_style']

folder = datasets_folders[1]
for f in glob.glob('/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author/*'):
    basename = f.split("/")[-1]
    txt_files = glob.glob(folder + "/*.txt")
    for file in txt_files:
        shutil.copy(file, f"/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author/{basename}/{basename}_Stories_AI_human_style/{file.split('/')[-1]}")
