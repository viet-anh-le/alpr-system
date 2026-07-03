from pathlib import Path

file_path = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Chuong/3_Cong_nghe.tex")
content = file_path.read_text()

# Section 3.3.5: Remove training section (lines 486-498), add OCR architecture flowchart
old = """\\textbf{Chi tiết huấn luyện}
Mô hình \\texttt{SmallLPR-Line-CTC} được huấn luyện bằng PyTorch Lightning trên tập ảnh crop biển số tại \\texttt{data/datasets/ocr}. Tập dữ liệu được chia thành hai phần: \\texttt{train} và \\texttt{valid}. Sau khi loại bỏ các ảnh quá nhỏ và các mẫu đã được đánh dấu lỗi trong quá trình rà soát dữ liệu, tập huấn luyện còn 27.434 ảnh, còn tập validation có 5.527 ảnh. Nhãn OCR được lấy trực tiếp từ tên file ảnh; đối với biển hai dòng, token \\texttt{[SEP]} được dùng để biểu diễn ranh giới giữa dòng trên và dòng dưới.

Trước khi đưa vào mô hình, mỗi ảnh được resize về kích thước \\(48 \\times 96\\) bằng cách giữ nguyên tỷ lệ và padding phần còn thiếu. Các ảnh có kích thước nhỏ hơn \\(20 \\times 8\\) pixel được loại bỏ để tránh đưa vào huấn luyện những mẫu không đủ thông tin ký tự. Giá trị pixel sau đó được chuẩn hóa theo công thức:
\\[
x' = (x - 127.5) \\times 0.0078125
\\]

Trong quá trình huấn luyện, các phép tăng cường dữ liệu được áp dụng để mô phỏng điều kiện thực tế của ảnh biển số trong video. Cụ thể, ảnh có thể được dịch chuyển, scale, xoay nhẹ, biến đổi phối cảnh, làm mờ chuyển động, làm mờ Gaussian, thêm nhiễu, thay đổi độ sáng, độ tương phản và sắc độ màu. Các phép biến đổi này giúp mô hình ổn định hơn trước sai số crop, rung hình, nhòe chuyển động và thay đổi ánh sáng trong môi trường giao thông.

Mô hình được huấn luyện trong 50 epoch, checkpoint tốt nhất được chọn theo độ chính xác trên tập validation. Quá trình tối ưu sử dụng AdamW với learning rate ban đầu \\(3 \\times 10^{-4}\\), weight decay \\(10^{-4}\\), batch size 64 và gradient clipping bằng 1.0. Learning rate được điều chỉnh bằng cosine annealing scheduler với giá trị nhỏ nhất \\(10^{-6}\\).

Trong mỗi epoch, mô hình được đánh giá bằng exact-match accuracy trên toàn bộ chuỗi biển số sau khi decode. Chỉ khi toàn bộ chuỗi dự đoán trùng khớp với nhãn, mẫu đó mới được tính là đúng.
\\section{Xử lí kết quả sau mô hình}"""

new = """\\paragraph{Flowchart kiến trúc OCR.}
Kiến trúc tổng quát của mô hình \\texttt{SmallLPR-Line-CTC} được mô tả bằng lưu đồ trong Hình~\\ref{fig:ocr_model_architecture}.

\\begin{figure}[H]
\\centering
% Placeholder for PlantUML diagram: OCR architecture flowchart
% Generate at: https://plantuml.com/plantuml
% File: docs/plantuml/ch3_ocr_architecture.puml
\\fbox{\\parbox{0.9\\textwidth}{\\centering\\small\\textbf{[Sơ đồ lưu đồ kiến trúc OCR -- xem file PlantUML: docs/plantuml/ch3\\_ocr\\_architecture.puml]}}}
\\caption{Lưu đồ kiến trúc mô hình OCR SmallLPR-Line-CTC}
\\label{fig:ocr_model_architecture}
\\end{figure}

\\section{Xử lí kết quả sau mô hình}"""

if old not in content:
    print("ERROR: old string not found for 3.3.5!")
    idx = content.find("\\textbf{Chi tiết huấn luyện}")
    if idx >= 0:
        print(f"Found at index {idx}")
        print(repr(content[idx:idx+200]))
else:
    content = content.replace(old, new)
    file_path.write_text(content)
    print("SUCCESS: replaced 3.3.5 (removed training, added OCR architecture flowchart)")
