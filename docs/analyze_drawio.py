import xml.etree.ElementTree as ET

tree = ET.parse('/home/vietanh/Documents/DATN/ALPR_Vietnamese/docs/ĐATN.drawio.xml')
root = tree.getroot()

for diagram in root.findall('diagram'):
    if diagram.get('name') == 'Pipeline-Overview':
        print(f"Diagram: {diagram.get('name')}")
        mxGraphModel = diagram.find('mxGraphModel')
        root_cell = mxGraphModel.find('root')
        for cell in root_cell.findall('mxCell'):
            cell_id = cell.get('id')
            value = cell.get('value', '')
            style = cell.get('style', '')
            geom = cell.find('mxGeometry')
            geom_str = ""
            if geom is not None:
                geom_str = f"x={geom.get('x', '0')} y={geom.get('y', '0')} w={geom.get('width', '0')} h={geom.get('height', '0')}"
            
            # truncate value and style for printing
            if len(value) > 30: value = value[:30] + '...'
            if len(style) > 30: style = style[:30] + '...'
            print(f"ID: {cell_id} | Value: {value} | Style: {style} | Geom: {geom_str}")

