import cv2
import numpy as np
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def send_email_alert(image_path):
    # Cấu hình email
    sender = "thaihoangnhat44@gmail.com"
    receiver = "hobadat2003@gmail.com"
    password = "veya uiht xcko qyej"  # Dùng app password nếu dùng Gmail 2 lớp
    subject = "Cảnh báo chuyển động"
    body = f"Phát hiện chuyển động! Xem ảnh tại: {image_path}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("Đã gửi email cảnh báo.")
    except Exception as e:
        print("Lỗi gửi email:", e)

def main():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    save_dir = "motion_images"
    os.makedirs(save_dir, exist_ok=True)
    sent_alert = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bright = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
        gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask = fgbg.apply(blur)
        kernel = np.ones((5,5),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print("Phát hiện chuyển động tại vị trí:", x, y, "kích thước:", w, h)
                motion_detected = True

        # Nếu phát hiện chuyển động, lưu ảnh và gửi email (chỉ gửi 1 lần cho mỗi sự kiện)
        if motion_detected and not sent_alert:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(save_dir, f"motion_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            send_email_alert(image_path)
            sent_alert = True
        elif not motion_detected:
            sent_alert = False  # Reset khi không còn chuyển động

        # Chuyển fgmask sang 3 kênh để ghép với frame màu
        mask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        # Ghép hai ảnh theo chiều ngang
        combined = np.hstack((frame, mask_color))
        # Hiển thị trên một cửa sổ duy nhất
        cv2.imshow('Motion Detection | Mask', combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()