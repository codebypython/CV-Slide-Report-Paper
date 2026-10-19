# Quy định Định dạng Toán học & Công thức (Math & Equation Formatting Rule)

## Nguyên nhân
Giao diện chat hiện tại không hỗ trợ bộ render LaTeX (KaTeX / MathJax). Khi sử dụng các ký tự LaTeX như `$ ... $`, `$$ ... $$`, `\le`, `\times`, `\sigma`, `\frac`, giao diện sẽ hiển thị lỗi font, ký tự rác hoặc mã thô khó đọc.

## Quy tắc bắt buộc đối với Agent:
1. **TUYỆT ĐỐI KHÔNG dùng cú pháp LaTeX trong câu trả lời:**
   - Không dùng dấu đô la `$ ... $` hoặc `$$ ... $$`.
   - Không dùng các lệnh gạch chéo ngược như `\le`, `\ge`, `\times`, `\sigma`, `\rho`, `\alpha`, `\beta`, `\frac{a}{b}`.

2. **Cách trình bày thay thế (100% rõ ràng trên mọi giao diện):**
   - **Ký hiệu số học & so sánh thông dụng:** Dùng ký tự Unicode trực tiếp: `<=`, `>=`, `+/-`, `x`, `^` (số mũ), `/` (chia).
     - Ví dụ: `MAE <= 4.25 tháng`, `102.5M tham số`, `384 x 384 pixels`.
   - **Tên tham số / biến số:** Viết dạng text rõ nghĩa: `alpha`, `beta`, `sigma`, `rho`, `R^2`, `y_true`, `y_pred`.
   - **Công thức nhiều dòng hoặc phép toán phức tạp:** Đặt trong khối code markdown (` ```text ` hoặc ` ```python `) với chú thích rõ ràng.
   - **Bảng số liệu:** Trình bày bằng Markdown Table tiêu chuẩn với số thập phân thông thường (ví dụ: `7.21 tháng`, `0.65`).
