from pathlib import Path

file_path = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Chuong/3_Cong_nghe.tex")
content = file_path.read_text()

# Section 3.4.1: Replace slot-aware correction text with flowchart placeholder
old_341 = """\\paragraph{Sửa lỗi ký tự mơ hồ theo slot}

Song song với bước kiểm tra định dạng, hệ thống còn thực hiện hiệu chỉnh các nhầm lẫn ký tự đặc trưng của OCR tiếng Việt. Nguyên nhân của những nhầm lẫn này là sự tương đồng hình dạng giữa một số cặp ký tự, ví dụ O/0, I/1, S/5, B/8, G/6, Z/2. Thay vì áp dụng một bảng thay thế toàn cục, hệ thống thực hiện hiệu chỉnh có nhận thức về ngữ cảnh vị trí (slot-aware): mỗi ký tự trong chuỗi đầu ra được căn chỉnh vào đúng vị trí slot tương ứng trong mẫu biển số, và việc sửa chỉ được thực hiện khi loại ký tự tại slot đó không khớp — ví dụ, slot D (digit) nhận chữ cái O sẽ được sửa thành 0, còn slot L (letter) nhận chữ số 0 sẽ được sửa thành O.

Quá trình căn chỉnh sử dụng giải thuật quy hoạch động với ba thao tác: khớp (align), bỏ qua token đầu vào (skip) và chèn token còn thiếu (missing), tương tự cơ chế edit-distance. Chi phí của từng thao tác được thiết kế bất đối xứng để ưu tiên các sửa đổi có chi phí thấp (chỉ cần đổi loại ký tự) hơn là xóa hay thêm ký tự. Sau căn chỉnh, mỗi sửa đổi làm giảm xác suất của ký tự đó đi một hệ số 0.92 để phản ánh sự không chắc chắn. Hệ thống thử căn chỉnh với tất cả 35 mẫu biển số và chọn mẫu có tổng chi phí căn chỉnh thấp nhất làm kết quả cuối cùng.

\\subsection{Tổng hợp đa khung hình}"""

new_341 = """\\paragraph{Thuật toán sửa lỗi ký tự theo slot.}
Song song với bước kiểm tra định dạng, hệ thống thực hiện hiệu chỉnh các nhầm lẫn ký tự đặc trưng của OCR tiếng Việt (O/0, I/1, S/5, B/8, G/6, Z/2). Thay vì bảng thay thế toàn cục, hệ thống sử dụng phương pháp \\textit{slot-aware}: mỗi ký tự được căn chỉnh vào đúng vị trí slot trong mẫu biển số, và sửa chỉ khi loại ký tự không khớp — ví dụ, slot D (digit) nhận chữ cái O sẽ được sửa thành 0.

Quá trình căn chỉnh sử dụng giải thuật quy hoạch động với ba thao tác: \\textbf{align} (khớp ký tự với vị trí mẫu), \\textbf{skip} (bỏ qua token thừa) và \\textbf{missing} (chèn token còn thiếu), tương tự cơ chế edit-distance. Chi phí bất đối xymmetric: đổi loại ký tự có chi phí thấp nhất, xóa/thêm ký tự có chi phí cao hơn. Sau căn chỉnh, mỗi sửa đổi giảm xác suất của ký tự đi hệ số 0.92. Hệ thống thử tất cả 35 mẫu biển số và chọn mẫu có tổng chi phí căn chỉnh thấp nhất.

\\paragraph{Flowchart thuật toán slot-aware correction.}
Quy trình căn chỉnh và sửa lỗi ký tự được mô tả bằng lưu đồ trong Hình~\\ref{fig:slot_correction_flow}.

\\begin{figure}[H]
\\centering
% Placeholder for PlantUML diagram: slot-aware correction flowchart
% Generate at: https://plantuml.com/plantuml
% File: docs/plantuml/ch3_slot_correction.puml
\\fbox{\\parbox{0.9\\textwidth}{\\centering\\small\\textbf{[Sơ đồ lưu đồ thuật toán slot-aware correction -- xem file PlantUML: docs/plantuml/ch3\\_slot\\_correction.puml]}}}
\\caption{Lưu đồ thuật toán căn chỉnh và sửa lỗi ký tự theo slot}
\\label{fig:slot_correction_flow}
\\end{figure}

\\subsection{Tổng hợp đa khung hình}"""

if old_341 not in content:
    print("ERROR: old string not found for 3.4.1!")
    idx = content.find("\\paragraph{Sửa lỗi ký tự mơ hồ theo slot}")
    if idx >= 0:
        print(f"Found at index {idx}")
        print(repr(content[idx:idx+100]))
else:
    content = content.replace(old_341, new_341)
    file_path.write_text(content)
    print("SUCCESS: replaced 3.4.1 slot-aware correction")
