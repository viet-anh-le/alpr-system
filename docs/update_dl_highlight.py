import xml.etree.ElementTree as ET

tree = ET.parse('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml')
xml_root = tree.getroot()

dl_style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontColor=#333333;fontSize=14;fontStyle=1;"

# IDs to update
dl_ids = ["txt_arr1", "txt_arr3", "txt_arr6"]

legend_cells = [
    {"id": "legend_box", "value": "", "style": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;", "x": "40", "y": "550", "w": "30", "h": "20"},
    {"id": "legend_text", "value": "Mô hình Deep Learning", "style": "text;html=1;align=left;verticalAlign=middle;fontSize=14;fontColor=#333333;fontStyle=2;", "x": "80", "y": "545", "w": "180", "h": "30"}
]

for diagram in xml_root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        mxGraphModel = diagram.find('mxGraphModel')
        root = mxGraphModel.find('root')
        
        for cell in root.findall('mxCell'):
            cell_id = cell.get('id')
            if cell_id in dl_ids:
                cell.set('style', dl_style)
                geom = cell.find('mxGeometry')
                if geom is not None:
                    # slightly increase height to accommodate the box
                    geom.set('height', '35')
                    
        # Add legend
        for cell_data in legend_cells:
            cell = ET.Element('mxCell', {'id': cell_data['id'], 'value': cell_data['value'], 'style': cell_data['style'], 'vertex': '1', 'parent': '1'})
            geom = ET.Element('mxGeometry', {'x': cell_data['x'], 'y': cell_data['y'], 'width': cell_data['w'], 'height': cell_data['h'], 'as': 'geometry'})
            cell.append(geom)
            root.append(cell)

tree.write('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml', encoding='UTF-8', xml_declaration=True)
