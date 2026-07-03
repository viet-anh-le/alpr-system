import xml.etree.ElementTree as ET

tree = ET.parse('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml')
xml_root = tree.getroot()

images = {
    "FPDbbJmA906ujo2PMi61-14": {"x": "40", "y": "55", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-15": {"x": "55", "y": "70", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-16": {"x": "70", "y": "85", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-18": {"x": "390", "y": "85", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-22": {"x": "710", "y": "70", "w": "100", "h": "160"},
    "FPDbbJmA906ujo2PMi61-26": {"x": "910", "y": "115", "w": "120", "h": "70"},
    "FPDbbJmA906ujo2PMi61-19": {"x": "1130", "y": "115", "w": "120", "h": "70"}
}

new_cells = [
    # Groups
    {"id": "bg_group1", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B3B3B3;strokeWidth=2;dashed=1;", "x": "20", "y": "20", "w": "1250", "h": "290"},
    {"id": "txt_group1", "value": "XỬ LÝ TỪNG KHUNG HÌNH", "style": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#333333;", "x": "30", "y": "30", "w": "300", "h": "30"},
    
    {"id": "bg_group2", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B3B3B3;strokeWidth=2;dashed=1;", "x": "195", "y": "340", "w": "800", "h": "190"},
    {"id": "txt_group2", "value": "TỔNG HỢP THEO TRACK (khi track kết thúc)", "style": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#333333;", "x": "205", "y": "350", "w": "400", "h": "30"},

    # Texts Row 1
    {"id": "txt_arr1", "value": "YOLOv5", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "300", "y": "105", "w": "80", "h": "30"},
    {"id": "txt_arr2", "value": "BoT-SORT", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "620", "y": "105", "w": "80", "h": "30"},
    {"id": "txt_arr3", "value": "YOLOv8-OBB", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "810", "y": "105", "w": "100", "h": "30"},
    {"id": "txt_arr4", "value": "Warp", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "1040", "y": "105", "w": "80", "h": "30"},

    # Arrows Row 1
    {"id": "arr1", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "310", "y": "135", "w": "60", "h": "30"},
    {"id": "arr2", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "630", "y": "135", "w": "60", "h": "30"},
    {"id": "arr3", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "830", "y": "135", "w": "60", "h": "30"},
    {"id": "arr4", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1050", "y": "135", "w": "60", "h": "30"},

    # Row 2
    {"id": "arr5", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=south;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1170", "y": "190", "w": "40", "h": "40"},
    {"id": "box_qr", "value": "Quality Router", "style": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;fontColor=#000000;fontSize=14;fontStyle=1;", "x": "1100", "y": "240", "w": "160", "h": "40"},
    {"id": "txt_arr6", "value": "OCR", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "1000", "y": "210", "w": "80", "h": "30"},
    {"id": "arr6", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1000", "y": "240", "w": "80", "h": "40"},
    {"id": "box_ocr", "value": "59-G1 509.58", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2A56A6;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "830", "y": "235", "w": "150", "h": "50"},

    # Row 3
    {"id": "arr7", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=south;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "885", "y": "290", "w": "40", "h": "60"},
    {"id": "txt_arr7", "value": "Tích lũy theo track_id", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "930", "y": "305", "w": "180", "h": "30"},
    
    {"id": "txt_buffer", "value": "Track Buffer", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "835", "y": "360", "w": "140", "h": "20"},
    {"id": "box_s1", "value": "59-G1 509.58", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2A56A6;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "835", "y": "390", "w": "140", "h": "30"},
    {"id": "box_s2", "value": "59-G? 509.58", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2A56A6;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "835", "y": "420", "w": "140", "h": "30"},
    {"id": "box_s3", "value": "5-G1 509.58", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2A56A6;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "835", "y": "450", "w": "140", "h": "30"},
    {"id": "box_s4", "value": "59-G1 509.58", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2A56A6;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "835", "y": "480", "w": "140", "h": "30"},

    {"id": "arr8", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "715", "y": "435", "w": "100", "h": "40"},
    {"id": "box_ctm", "value": "CTM + Voting", "style": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;fontColor=#000000;fontSize=14;fontStyle=1;", "x": "535", "y": "425", "w": "160", "h": "60"},
    
    {"id": "arr9", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "415", "y": "435", "w": "100", "h": "40"},
    {"id": "box_final", "value": "59-G1 509.58", "style": "rounded=1;whiteSpace=wrap;html=1;fillColor=#4A86E8;strokeColor=none;fontColor=#FFFFFF;fontSize=20;fontStyle=1;", "x": "215", "y": "415", "w": "180", "h": "80"},

    # Caption
    {"id": "caption", "value": "Fig. 1: ALPR pipeline", "style": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=22;fontFamily=Times New Roman;", "x": "450", "y": "550", "w": "400", "h": "40"}
]

for diagram in xml_root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        mxGraphModel = diagram.find('mxGraphModel')
        root = mxGraphModel.find('root')
        
        saved_nodes = []
        for child in list(root):
            if child.get('id') in ['0', '1']:
                continue
            if child.get('id') in images:
                child.set('parent', '1')
                geom = child.find('mxGeometry')
                if geom is not None:
                    geom.set('x', images[child.get('id')]['x'])
                    geom.set('y', images[child.get('id')]['y'])
                    geom.set('width', images[child.get('id')]['w'])
                    geom.set('height', images[child.get('id')]['h'])
                saved_nodes.append(child)
            root.remove(child)
            
        for cell_data in new_cells:
            cell = ET.Element('mxCell', {'id': cell_data['id'], 'value': cell_data['value'], 'style': cell_data['style'], 'vertex': '1', 'parent': '1'})
            geom = ET.Element('mxGeometry', {'x': cell_data['x'], 'y': cell_data['y'], 'width': cell_data['w'], 'height': cell_data['h'], 'as': 'geometry'})
            cell.append(geom)
            root.append(cell)
            
        for node in saved_nodes:
            # We want images to be drawn ON TOP of the background boxes, so append them after
            root.append(node)

# To ensure the background boxes are behind everything, we must insert them at the beginning of the root element (index 2, after '0' and '1')
# Let's adjust the order inside the root.
for diagram in xml_root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        mxGraphModel = diagram.find('mxGraphModel')
        root = mxGraphModel.find('root')
        elements = list(root)
        new_order = []
        bg_elements = []
        other_elements = []
        for el in elements:
            if el.get('id') in ['0', '1']:
                new_order.append(el)
            elif el.get('id') in ['bg_group1', 'bg_group2']:
                bg_elements.append(el)
            else:
                other_elements.append(el)
        
        # Clear root and re-append in order
        for el in list(root):
            root.remove(el)
        for el in new_order + bg_elements + other_elements:
            root.append(el)

tree.write('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml', encoding='UTF-8', xml_declaration=True)
