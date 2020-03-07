import xml.etree.ElementTree as ET
import ipdb

path ='half-nut-template.xml'

tree = ET.parse(path)
root = tree.getroot()

cnt =0
for child in root[1][0]:
    cnt+=1
    if cnt ==1:
        continue

    size_vec = child.attrib['size'].split(' ')

    new_size= [float(i)*0.95 for i in size_vec]
    new_str = [str(i) for i in new_size]
    new_str = ' '.join(new_str)
    print('new_size: ', new_str)
    child.attrib['size'] = new_str

    size_vec = child.attrib['pos'].split(' ')
    new_size = [float(i)*0.95  for i in size_vec]
    new_str = [str(i) for i in new_size]
    new_str = ' '.join(new_str)
    print('new pos: ', new_str)
    child.attrib['pos'] = new_str

tree.write('half-nut.xml')
