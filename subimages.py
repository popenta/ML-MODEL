import os
from xml.etree import ElementTree

base_train = "train"
base_val = "validation"

all_xml_files = []

for file in os.listdir(base_train):
    if file.endswith(".xml"):
        all_xml_files.append(file)

#pt pozele de antrenament
os.chdir(base_train)

for file in all_xml_files:
    img_boxes = []
    
    tree = ElementTree.parse(file)
    root = tree.getroot()

    for object in root.findall("object"):
        tip = object.find("name")
        
        for box in object.find("bndbox"):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

        