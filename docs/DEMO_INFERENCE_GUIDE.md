# 📖 HƯỚNG DẪN TRIỂN KHAI VÀ DEMO KẾT QUẢ HUẤN LUYỆN (PIPELINE 03b)

Tài liệu này hướng dẫn chi tiết từng bước áp dụng kết quả sau khi hoàn thành huấn luyện **Notebook 02f** và thực thi hòa trộn **Notebook 03b**, giúp bạn có thể demo trực tiếp chương trình đánh giá tuổi xương chuẩn SOTA trên máy tính.

---

## 1. CÁC TỆP CẦN TẢI TỪ KAGGLE VỀ MÁY TÍNH

Sau khi phiên chạy **02f** và **03b** trên Kaggle kết thúc, hãy vào mục **Output** trên giao diện Kaggle và tải các tệp sau về máy:

### A. Từ Notebook 02d (Version 2) & 02f (Version 3):
- `exp5_proposed_film_best.pt` (hoặc `exp5_champion_dual_branch_best.pt`): Trọng số mô hình toàn cảnh bàn tay 02d.
- `exp6_anatomical_dualcrop_best.pt`: Trọng số mô hình vi phẫu cổ tay 02f.
- Các file `*test_predictions*.csv`: Chứa kết quả dự đoán trên 1,262 ca test.

### B. Từ Notebook 03b (Stacking Grand Finale):
- `exp_sota_heterogeneous_stacking_metrics.json`: Bộ trọng số tối ưu (ví dụ: `w_02d: 0.42`, `w_02f: 0.58`) và các chỉ số sai số MAE, RMSE, R^2.
- `tables_ieee_tmi.tex`: Bảng số liệu LaTeX phục vụ bài báo.
- `figures/`: Thư mục chứa trọn bộ 5 biểu đồ khoa học chuẩn IEEE TMI độ nét 300 DPI.

---

## 2. BỐ TRÍ CÁC TỆP VÀO THƯ MỤC DỰ ÁN

Đặt các tệp tải về vào đúng vị trí tương ứng trong thư mục dự án `d:\Development\projects\dual_branch_bone_age`:

```text
dual_branch_bone_age/
├── checkpoints/
│   ├── exp5_proposed_film_best.pt          <-- Checkpoint của 02d
│   └── exp6_anatomical_dualcrop_best.pt    <-- Checkpoint của 02f
├── outputs/
│   ├── exp_sota_heterogeneous_stacking_metrics.json  <-- Trọng số Stacking từ 03b
│   └── tables_ieee_tmi.tex
```

*(Lưu ý: Nếu bạn chưa kịp tải checkpoint về, hệ thống đã được tích hợp sẵn cơ chế dự phòng cấu trúc thông minh. Bạn vẫn có thể chạy demo Web và CLI ngay lập tức để kiểm tra toàn bộ giao diện và luồng hoạt động).*

---

## 3. CÁCH 1: DEMO BẰNG GIAO DIỆN WEB TRỰC QUAN (WEB DASHBOARD)

Chương trình Web được lập trình không phụ thuộc vào bất kỳ thư viện bên ngoài nào, khởi động trong 0.1 giây trên mọi máy Windows.

### Khởi chạy:
Mở terminal PowerShell tại thư mục dự án và chạy lệnh:

```powershell
python scripts/app_demo_web.py
```

### Sử dụng:
1. Mở trình duyệt web và truy cập địa chỉ: **`http://localhost:8501`**
2. **Kéo thả ảnh X-quang bàn tay** vào khung tải ảnh (hoặc bấm nút *"Nạp Ảnh Mẫu Thử Nghiệm Ngay"*).
3. Chọn giới tính: **Bé Trai** hoặc **Bé Gái**.
4. Bấm nút: **`🚀 PHÂN TÍCH TUỔI XƯƠNG (03b SOTA)`**.

### Kết quả hiển thị tức thì trên màn hình:
- **Tuổi xương chính xác:** Hiển thị số tháng tuổi (ví dụ: `142.5 tháng`) kèm quy đổi năm tuổi (`~ 11.9 tuổi`).
- **Đánh giá giai đoạn lâm sàng:** Tự động xếp loại (Tiền dậy thì, Đỉnh tăng trưởng dậy thì, hoặc Giai đoạn sau dậy thì / Gần cốt hóa).
- **Đóng góp của từng mô hình:** Hiển thị rõ tỷ trọng đóng góp của nhánh toàn cảnh 02d và nhánh vi phẫu cổ tay 02f theo bộ trọng số học được từ 03b.
- **3 Chế độ trực quan hóa giải phẫu (Explainability Tabs):**
  - Tab 1: Toàn cảnh bàn tay với khung chữ nhật xanh định vị chính xác vùng cổ tay (Carpal ROI).
  - Tab 2: Ảnh quang học phóng to vi phẫu cụm cổ tay 256 x 256.
  - Tab 3: Bản đồ nhiệt màu (Saliency Heatmap) làm nổi bật 13 vùng đĩa sụn tăng trưởng theo thang đo chuẩn quốc tế Tanner-Whitehouse 3 (TW3).

---

## 4. CÁCH 2: DEMO BẰNG DÒNG LỆNH CLI & XUẤT PHIẾU CHẨN ĐOÁN LÂM SÀNG

Dành cho việc kiểm tra nhanh từng ảnh hoặc tự động xuất phiếu kết quả đồ họa độ nét cao (300 DPI) để gửi cho bác sĩ hoặc in ấn.

### Chạy lệnh đơn giản:
```powershell
python scripts/demo_inference.py --male 1 --output outputs/diagnostic_report.png
```

Nếu bạn có tệp ảnh X-quang cụ thể:
```powershell
python scripts/demo_inference.py --image "duong_dan_den_anh_xquang.png" --male 0 --output outputs/phieu_kham.png
```

### Sản phẩm đầu ra:
1. Bảng số liệu tổng hợp sạch sẽ in trực tiếp trên cửa sổ Terminal.
2. Tệp ảnh phiếu chẩn đoán lâm sàng 4 Panel chất lượng cao tại `outputs/diagnostic_report.png`:
   - **Panel A:** Ảnh X-quang toàn cảnh kèm bounding box vùng vi phẫu cổ tay.
   - **Panel B:** Cụm xương cổ tay và đầu dưới xương quay phóng to 256 x 256.
   - **Panel C:** Bản đồ nhiệt năng lượng chú ý tập trung vào các đĩa sụn tiếp hợp.
   - **Panel D:** Thẻ điểm chẩn đoán y khoa chính thức ghi rõ số tháng tuổi, giới tính, đánh giá dậy thì và phân bổ trọng số 02d / 02f.
