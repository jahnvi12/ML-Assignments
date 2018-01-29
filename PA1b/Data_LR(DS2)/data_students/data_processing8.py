from PIL import Image
import glob

def features(data):
    rows = len(data)
    cols = len(data[0])
    all_features = []
    for j in xrange(cols):
        bins = [0]*32
        for i in xrange(rows):
            bins[data[i][j]/8] += 1
        all_features.extend(bins)
    return all_features

def write_d(file_name, data):
    f = open(file_name, "w")
    f.write(header)
    f.write("%d %d\n"%(len(data),len(data[0])))
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
            f.write("%d "%data[i][j])
        f.write("\n")

splits = ["Test", "Train"]
classes = ["mountain", "forest","insidecity","coast"]
labels = {}
labels["mountain"] = [1,0,0,0]
labels["forest"] = [0,1,0,0]
labels["insidecity"] = [0,0,1,0]
labels["coast"] = [0,0,0,1]

header = "%%MatrixMarket matrix array real general\n"
for split in splits:
    all_features = []
    class_labels = []
    features_file =  split + "_features8"
    labels_file = split + "_labels8"
    for clas in classes:
        for filename in glob.glob(clas + "/" + split + '/*.jpg'): #assuming gif
            im=Image.open(filename)
            data = im.getdata()
            t = features(data)
            t.extend(labels[clas])
            all_features.append(t)
            
            #class_labels.append([labels[clas]])
    write_d(features_file, all_features)
    #write_d(labels_file, class_labels)
