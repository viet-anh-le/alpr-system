from pathlib import Path

file_path = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Chuong/3_Cong_nghe.tex")
content = file_path.read_text()

old = """Mô hình học sâu được huấn luyện theo bài toán phân loại ảnh nhẹ, sử dụng backbone YOLOv8n-cls khởi tạo từ trọng số đã huấn luyện trước. Ảnh đầu vào được resize về kích thước nhỏ để giữ tốc độ suy luận phù hợp với pipeline video. Trong quá trình huấn luyện, các phép tăng cường như thay đổi màu sắc, dịch chuyển, co giãn, lật ngang, auto-augment và random erasing được sử dụng để mô phỏng biến động ánh sáng, sai số crop và suy giảm cục bộ của biển số trong video thực tế. Bộ tối ưu, momentum, weight decay, AMP và seed được giữ nhất quán với các thí nghiệm YOLO khác để dễ tái lập.

\subsection{Kiến trúc và huấn luyện mô hình OCR (SmallLPR-Line-CTC)}"""

new = """\paragraph{Flowchart thuật toán Quality Router.}
Quy trình phân loại chất lượng và định tuyến xử lý được mô tả bằng lưu đồ trong Hình~\ref{fig:quality_router_flow}.

\begin{figure}[H]
\centering
% Placeholder for PlantUML diagram: quality router flowchart
% Generate at: https://plantuml.com/plantuml
% File: docs/plantuml/ch3_quality_router.puml
\fbox{\parbox{0.9\textwidth}{\centering\small\textbf{[Sơ đồ lưu đồ Quality Router -- xem file PlantUML: docs/plantuml/ch3\_quality\_router.puml]}}}
\caption{Lưu đồ thuật toán phân loại chất lượng và định tuyến xử lý}
\label{fig:quality_router_flow}
\end{figure}

\subsection{Kiến trúc và thiết kế mô hình OCR (SmallLPR-Line-CTC)}"""

if old not in content:
    print("ERROR: old string not found for 3.3.4!")
    idx = content.find("Mô hình học sâu được huấn luyện theo bài toán phân loại")
    if idx >= 0:
        print(f"Found at index {idx}")
        print(repr(content[idx:idx+200]))
else:
    content = content.replace(old, new)
    file_path.write_text(content)
    print("SUCCESS: replaced 3.3.4")
