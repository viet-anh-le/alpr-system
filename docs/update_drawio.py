import xml.etree.ElementTree as ET

tree = ET.parse('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml')
xml_root = tree.getroot()

images = {
    "FPDbbJmA906ujo2PMi61-14": {"x": "40", "y": "70", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-15": {"x": "55", "y": "85", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-16": {"x": "70", "y": "100", "w": "220", "h": "130"},
    "FPDbbJmA906ujo2PMi61-18": {"x": "450", "y": "90", "w": "240", "h": "140"},
    "FPDbbJmA906ujo2PMi61-19": {"x": "860", "y": "40", "w": "140", "h": "80"},
    "FPDbbJmA906ujo2PMi61-22": {"x": "860", "y": "140", "w": "120", "h": "190"},
    "FPDbbJmA906ujo2PMi61-26": {"x": "1140", "y": "340", "w": "130", "h": "70"}
}

new_cells = [
    # Text nodes
    {"id": "text_arrow1", "value": "Vehicle &amp; LP&lt;br&gt;detection", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "320", "y": "80", "w": "100", "h": "40"},
    {"id": "text_arrow2", "value": "Vehicle &amp; LP&lt;br&gt;filtering", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "700", "y": "80", "w": "120", "h": "40"},
    {"id": "text_arrow3", "value": "Rotate and update&lt;br&gt;rotation angle", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "1050", "y": "170", "w": "160", "h": "40"},
    {"id": "text_arrow4", "value": "Character&lt;br&gt;Detection", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "1020", "y": "300", "w": "100", "h": "40"},
    {"id": "text_arrow5", "value": "Store detection&lt;br&gt;information", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "740", "y": "300", "w": "140", "h": "40"},
    {"id": "text_arrow6", "value": "Character time-series&lt;br&gt;matching", "style": "text;html=1;align=center;verticalAlign=middle;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "480", "y": "300", "w": "140", "h": "40"},

    # Arrows
    {"id": "arr1", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "320", "y": "130", "w": "100", "h": "40"},
    
    # Arrow 2 Fork
    {"id": "arr2_base", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "710", "y": "130", "w": "50", "h": "40"},
    {"id": "arr2_up_down", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "740", "y": "60", "w": "40", "h": "170"},
    {"id": "arr2_top", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "780", "y": "40", "w": "60", "h": "80"},
    {"id": "arr2_bot", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "780", "y": "170", "w": "60", "h": "80"},
    
    # Arrow 3 Elbow
    {"id": "arr3_horiz", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1000", "y": "210", "w": "40", "h": "40"},
    {"id": "arr3_vert", "value": "", "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1000", "y": "210", "w": "40", "h": "160"},
    {"id": "arr3_right", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1040", "y": "330", "w": "80", "h": "80"},
    
    # Reverse Arrows
    {"id": "arr4", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "1020", "y": "355", "w": "100", "h": "40"},
    {"id": "arr5", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "760", "y": "355", "w": "100", "h": "40"},
    {"id": "arr6", "value": "", "style": "shape=mxgraph.arrows2.arrow;dy=0.6;dx=40;direction=west;notch=0;fillColor=#5A89D6;strokeColor=#2A56A6;", "x": "500", "y": "355", "w": "100", "h": "40"},
    
    # Boxes
    {"id": "box_aoi_1", "value": "AOI-1778", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=#2A56A6;fillColor=#FFFFFF;fontColor=#2A56A6;fontSize=16;fontStyle=1;", "x": "890", "y": "355", "w": "100", "h": "40"},
    {"id": "box_s1", "value": "AOI-1778", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=#2A56A6;fillColor=#FFFFFF;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "630", "y": "355", "w": "100", "h": "30"},
    {"id": "box_s2", "value": "AOI-177", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=#2A56A6;fillColor=#FFFFFF;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "630", "y": "385", "w": "100", "h": "30"},
    {"id": "box_s3", "value": "AOI-1778", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=#2A56A6;fillColor=#FFFFFF;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "630", "y": "415", "w": "100", "h": "30"},
    {"id": "box_s4", "value": "AOI-1 78", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=#2A56A6;fillColor=#FFFFFF;fontColor=#2A56A6;fontSize=14;fontStyle=1;", "x": "630", "y": "445", "w": "100", "h": "30"},
    
    # Final Box
    {"id": "box_final", "value": "AOI-1778", "style": "rounded=0;whiteSpace=wrap;html=1;strokeColor=none;fillColor=#4A86E8;fontColor=#FFFFFF;fontSize=20;fontStyle=1;", "x": "330", "y": "335", "w": "140", "h": "80"},
    
    # Caption
    {"id": "caption", "value": "Fig. 1: ALPR pipeline", "style": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=22;fontFamily=Times New Roman;", "x": "450", "y": "520", "w": "400", "h": "40"}
]

for diagram in xml_root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        mxGraphModel = diagram.find('mxGraphModel')
        root = mxGraphModel.find('root')
        
        # Save image nodes
        saved_nodes = []
        # Need to deep search because images might be in a group
        def extract_nodes(node):
            for child in node.findall('mxCell'):
                if child.get('id') in images:
                    # Update parent to '1' and geometry
                    child.set('parent', '1')
                    geom = child.find('mxGeometry')
                    if geom is not None:
                        geom.set('x', images[child.get('id')]['x'])
                        geom.set('y', images[child.get('id')]['y'])
                        geom.set('width', images[child.get('id')]['w'])
                        geom.set('height', images[child.get('id')]['h'])
                    saved_nodes.append(child)
                # Recurse if there are children? mxCells don't have mxCell children in raw drawio XML, they are all flat under root!
                # Wait, yes, in flat drawio XML, groups just set parent=groupID. The elements are still direct children of root.
        
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
            # Remove all elements except '0' and '1'
            root.remove(child)
            
        # Append saved nodes back to root
        for node in saved_nodes:
            root.append(node)
            
        # Append new nodes
        for cell_data in new_cells:
            cell = ET.Element('mxCell', {'id': cell_data['id'], 'value': cell_data['value'], 'style': cell_data['style'], 'vertex': '1', 'parent': '1'})
            geom = ET.Element('mxGeometry', {'x': cell_data['x'], 'y': cell_data['y'], 'width': cell_data['w'], 'height': cell_data['h'], 'as': 'geometry'})
            cell.append(geom)
            root.append(cell)

tree.write('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml', encoding='UTF-8', xml_declaration=True)
