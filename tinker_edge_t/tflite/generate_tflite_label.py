import json

if __name__ == "__main__":
    
    with open("label_map.txt", "r") as readFile:
        dict_ = json.load(readFile)

    with open("tflite_label_map.txt", "w") as writeFile:
        for key in dict_.keys():
            writeFile.write("{} {} \n".format(key, dict_[key]))
            
