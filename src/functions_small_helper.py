import os
from matplotlib import pyplot as plt
import seaborn as sns


# paths, project structure
def create_folder_if_not_exists(path_to_folder):
    if not os.path.exists(path_to_folder):
        print(f"<{path_to_folder}> does not yet exist -> create local folder...")
        os.mkdir(path_to_folder)
    print(f"<{path_to_folder}> exists: ", os.path.exists(path_to_folder))




# dataframes
def get_sorted_labels_of_column(df, column_name):
    return sorted(df[column_name].unique())


# plotting
def plot_column_count(df, column_name, bar_labels=True, rotate_xticks=False, order_function=get_sorted_labels_of_column, figsize=(10,5), title_suffix="", fontsize=10):
    plt.figure(figsize=figsize)
    ax = sns.countplot(df, x=column_name, order=order_function(df, column_name))
    if bar_labels:
        plt.bar_label(ax.containers[0], fontsize=fontsize, rotation=90, padding=2)
    if rotate_xticks:
        plt.xticks(rotation=90, fontsize=fontsize)
    plt.title(f"Counts per {column_name}" + title_suffix)
    plt.show()