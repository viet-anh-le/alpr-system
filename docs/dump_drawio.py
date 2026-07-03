import xml.etree.ElementTree as ET
import json

tree = ET.parse('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml')
root = tree.getroot()

cells_data = []

for diagram in root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        mxGraphModel = diagram.find('mxGraphModel')
        root_cell = mxGraphModel.find('root')
        for cell in root_cell.findall('mxCell'):
            c_dict = {
                'id': cell.get('id'),
                'parent': cell.get('parent'),
                'value': cell.get('value', ''),
                'style': cell.get('style', '')
            }
            geom = cell.find('mxGeometry')
            if geom is not None:
                c_dict['geom'] = {
                    'x': geom.get('x'),
                    'y': geom.get('y'),
                    'width': geom.get('width'),
                    'height': geom.get('height'),
                    'as': geom.get('as')
                }
            cells_data.append(c_dict)

with open('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/cells.json', 'w') as f:
    json.dump(cells_data, f, indent=2, ensure_ascii=False)

