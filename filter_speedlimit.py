import os
import xml.etree.ElementTree as ET

# annotations 폴더 경로 설정
annotations_dir = 'C:/Users/2019124074/traffic-sign/kaggle_dataset/annotations'

# 각 파일의 태그를 확인하는 함수
def check_tags_in_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    print(f"파일 이름: {os.path.basename(file_path)}")
    for obj in root.findall('object'):
        name = obj.find('name').text
        print(f"태그: {name}")
    print("-----")

# annotations 폴더 내의 모든 XML 파일의 태그 확인
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith('.xml'):
        file_path = os.path.join(annotations_dir, annotation_file)
        check_tags_in_xml(file_path)
