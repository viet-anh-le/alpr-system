from pathlib import Path

file_path = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Chuong/3_Cong_nghe.tex")
content = file_path.read_text()

# Find the CTM voting tikzpicture start (the big input node)
ctm_tikz_start = content.find("\\node[input] (inp) {")
print(f"CTM tikz start: {ctm_tikz_start}")

# Find the CTM figure end (after \end{figure})
ctm_fig_end = content.find("\\end{figure}", content.find("\\caption{Quy trình tổng hợp"))
print(f"CTM figure end: {ctm_fig_end}")

# Find the quality router figure
qr_start = content.find("\\begin{figure}[H]", ctm_fig_end)
print(f"QR figure start: {qr_fig_start := qr_start}")

qr_end = content.find("\\end{figure}", qr_start)
print(f"QR figure end: {qr_end}")

# Find \end{document} after QR figure
end_doc = content.find("\\end{document}", qr_end)
print(f"end document: {end_doc}")

# Build the replacement text
ctm_placeholder = """\\paragraph{Flowchart thuật toán CTM voting.}
Quy trình tổng hợp đa khung hình CTM (Character Time-series Matching) được mô tả bằng lưu đồ trong Hình~\\ref{fig:ctm_voting_flowchart}.

\\begin{figure}[H]
\\centering
% Placeholder for PlantUML diagram: CTM voting flowchart
% Generate at: https://plantuml.com/plantuml
% File: docs/plantuml/ch3_ctm_voting.puml
\\fbox{\\parbox{0.9\\textwidth}{\\centering\\small\\textbf{[Sơ đồ lưu đồ CTM voting -- xem file PlantUML: docs/plantuml/ch3\\_ctm\\_voting.puml]}}}
\\caption{Quy trình tổng hợp đa khung hình CTM (Character Time-series Matching)}
\\label{fig:ctm_voting_flowchart}
\\end{figure}

"""

qr_placeholder = """\\begin{figure}[H]
\\centering
% Placeholder for PlantUML diagram: quality router flowchart
% Generate at: https://plantuml.com/plantuml
% File: docs/plantuml/ch3_quality_router.puml
\\fbox{\\parbox{0.9\\textwidth}{\\centering\\small\\textbf{[Sơ đồ lưu đồ Quality Router -- xem file PlantUML: docs/plantuml/ch3\\_quality\\_router.puml]}}}
\\caption{Luồng xử lý sau Quality Router -- định tuyến 3 hướng và tổng hợp đa khung hình}
\\label{fig:quality_router_flow}
\\end{figure}


\\end{document}"""

# Replace: everything from CTM tikz start to QR figure end, then add \end{document}
before = content[:ctm_tikz_start]
after_end = content[end_doc:]  # includes \end{document}

new_content = before + ctm_placeholder + qr_placeholder + after_end

file_path.write_text(new_content)
print(f"SUCCESS: replaced CTM + QR figures")
print(f"Old length: {len(content)}, New length: {len(new_content)}")
