import glob
import os

files = glob.glob('Chuong/*.tex')
for file in files:
    with open(file, 'r') as f:
        content = f.read()
    
    new_content = content.replace('\\begin{figure}[H]', '\\begin{figure}[htbp]')
    new_content = new_content.replace('\\begin{table}[H]', '\\begin{table}[htbp]')
    
    if new_content != content:
        with open(file, 'w') as f:
            f.write(new_content)
        print(f"Updated {file}")
