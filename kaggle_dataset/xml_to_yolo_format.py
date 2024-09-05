import os
import xml.etree.ElementTree as ET

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_xml_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    filename = root.find('filename').text
    out_file = open(os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt'), 'w')
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in ['sign', 'speedlimit'] or int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_coordinates((w,h), b)
        out_file.write("0" + " " + " ".join([str(a) for a in bb]) + '\n')
    
    out_file.close()

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(input_dir, xml_file), output_dir)

# 사용 예시
input_dir = './annotations'
output_dir = './yolo'
process_folder(input_dir, output_dir)