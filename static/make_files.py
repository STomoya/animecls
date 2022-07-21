
import glob
import os
import pickle

def imagenet_sketch():
    with open('./static/imagenet_folder2labels.txt', 'r') as fin:
        lines = fin.read().strip().split('\n')
    lines = [line.split(' ') for line in lines]
    folders, labels_plusone, names = list(zip(*lines))
    folder2label = {folder: int(label)-1 for folder, label in zip(folders, labels_plusone)}
    label2name = {int(label)-1: name for label, name in zip(labels_plusone, names)}

    with open('./static/imagenet_sketch/folder2label.pickle', 'wb') as fout:
        pickle.dump(folder2label, fout)
    with open('./static/imagenet_sketch/label2name.pickle', 'wb') as fout:
        pickle.dump(label2name, fout)

def animeface(root='./data/animeface'):
    # extract folders from file names because some folders are empty
    files = glob.glob(os.path.join(root, '**', '*'), recursive=True)
    files = [file for file in files if os.path.isfile(file) and file.endswith('.png')]
    folders = sorted(list(set([file.split('/')[-2] for file in files])))
    folder2label = {folder: label for label, folder in enumerate(folders)}
    label2name = {label: '_'.join(folder.split('_')[1:]) for folder, label in folder2label.items()}

    with open('./static/animeface/folder2label.pickle', 'wb') as fout:
        pickle.dump(folder2label, fout)
    with open('./static/animeface/label2name.pickle', 'wb') as fout:
        pickle.dump(label2name, fout)

if __name__=='__main__':
    imagenet_sketch()
    animeface()
