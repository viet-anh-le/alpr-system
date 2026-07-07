#!/bin/bash

# Thư mục nguồn (thư mục hiện tại)
SOURCE_DIR="$(pwd)"

# Thư mục đích (đặt ngang hàng với thư mục hiện tại)
DEST_DIR="../ALPR_Vietnamese_Copy"

echo "Bắt đầu tạo bản sao từ: $SOURCE_DIR"
echo "Lưu tới đích: $DEST_DIR"

# Tạo thư mục đích nếu chưa có
mkdir -p "$DEST_DIR"

# Sử dụng rsync kết hợp bộ lọc .gitignore
# -a: Giữ nguyên mọi quyền (permissions), timestamp, symlinks...
# -v: Hiển thị log (verbose)
# --filter=":- .gitignore": Đọc và áp dụng luật loại bỏ từ TẤT CẢ các file .gitignore (kể cả ở các thư mục con)
# --exclude=".git": Không copy thư mục lịch sử git (nếu bạn muốn copy cả .git thì xóa dòng này đi)
rsync -av \
    --filter=":- .gitignore" \
    --exclude=".git" \
    "$SOURCE_DIR/" "$DEST_DIR/"

echo "======================================"
echo "✅ Đã tạo bản sao thành công tại: $(realpath "$DEST_DIR")"
