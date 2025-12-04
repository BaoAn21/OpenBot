import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def create_dummy_image():
    """
    Tạo một ảnh đen với các đường trắng giả lập:
    - 2 đường gần như thẳng đứng (cần giữ lại - Line Tracking)
    - 1 đường ngang và 1 đường chéo (cần lọc bỏ - Nhiễu)
    """
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Đường thẳng đứng 1 (Hơi nghiêng nhẹ mô phỏng thực tế)
    cv2.line(img, (200, 50), (220, 550), (255, 255, 255), 5)
    
    # Đường thẳng đứng 2 (Bên phải)
    cv2.line(img, (400, 50), (380, 550), (255, 255, 255), 5)
    
    # Đường ngang (Nhiễu)
    cv2.line(img, (50, 300), (550, 300), (255, 255, 255), 5)
    
    # Đường chéo (Nhiễu)
    cv2.line(img, (100, 100), (500, 500), (255, 255, 255), 5)
    
    return img

def calculate_angle(x1, y1, x2, y2):
    """Tính góc của đường thẳng so với trục hoành (trục X) theo độ"""
    # atan2 trả về radian từ -pi đến pi
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def process_line_tracking():
    # 1. Tạo ảnh đầu vào
    original_img = create_dummy_image()
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Edge Detection (Canny)
    # Ngưỡng 50, 150 là ví dụ, thực tế cần tinh chỉnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 3. Hough Transform (Probabilistic)
    # rho=1, theta=np.pi/180 (độ phân giải 1 độ), threshold=50 (số điểm vote tối thiểu)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Tạo 2 bản sao ảnh để vẽ kết quả
    img_all_lines = original_img.copy()
    img_filtered_lines = original_img.copy()

    # Danh sách để debug
    print(f"Tổng số đường tìm thấy: {len(lines) if lines is not None else 0}")
    max_len = 0           # Lưu độ dài lớn nhất tìm được
    best_line_coords = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính góc
            angle = calculate_angle(x1, y1, x2, y2)
            
            # Vẽ TẤT CẢ các line tìm được (Màu ĐỎ) vào hình 1
            cv2.line(img_all_lines, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # --- LOGIC LỌC (FILTERING) ---
            # Phương đứng thì góc so với trục X phải gần 90 độ hoặc -90 độ
            # Ta chấp nhận sai số +/- 30 độ (tức là từ 60 đến 120, hoặc -120 đến -60)
            is_vertical = False
            
            # Chuyển góc về dương để dễ so sánh (0 đến 180)
            abs_angle = abs(angle)
            
            # Ngưỡng góc chấp nhận: > 60 độ (gần đứng)
            if abs_angle > 60 and abs_angle < 120:
                is_vertical = True
            
            
            if is_vertical:
                # tinh do dai
                current_len = math.hypot(x2 - x1, y2 - y1)
                if current_len > max_len:
                    max_len = current_len       # Cập nhật độ dài kỷ lục
                    best_line_coords = (x1, y1, x2, y2)
                
                # Vẽ những đường ĐẠT CHUẨN (Màu XANH LÁ) vào hình 2
                cv2.line(img_filtered_lines, (x1, y1), (x2, y2), (0, 255, 0), 5)
                print(f"Giữ lại line: ({x1},{y1})->({x2},{y2}), Góc: {angle:.2f} -> Vertical, dai {current_len}")
            else:
                print(f"Loại bỏ line: ({x1},{y1})->({x2},{y2}), Góc: {angle:.2f} -> Horizontal/Diagonal")
    if best_line_coords is not None:
        bx1, by1, bx2, by2 = best_line_coords
        
        # Vẽ đường dài nhất đè lên (Màu ĐỎ, dày hơn) để highlight
        cv2.line(img_filtered_lines, (bx1, by1), (bx2, by2), (0, 0, 255), 5)
        
        print(f"Đường dẫn hướng (Dài nhất): Độ dài {max_len:.2f} ({bx1},{by1})->({bx2},{by2})")
    # 4. Hiển thị kết quả bằng Matplotlib
    plt.figure(figsize=(15, 5))

    # Cột 1: Ảnh gốc + Canny
    plt.subplot(1, 3, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('1. Canny Edge Detection')
    plt.axis('off')

    # Cột 2: Hough Transform (Raw)
    plt.subplot(1, 3, 2)
    # Convert BGR to RGB để hiển thị đúng màu trên Matplotlib
    plt.imshow(cv2.cvtColor(img_all_lines, cv2.COLOR_BGR2RGB))
    plt.title('2. Hough Transform (Raw Lines - Red)')
    plt.axis('off')

    # Cột 3: Kết quả sau khi lọc phương
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_filtered_lines, cv2.COLOR_BGR2RGB))
    plt.title('3. Filtered Vertical Lines (Green)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_line_tracking()