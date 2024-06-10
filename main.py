import cv2
import torch
import yaml
import numpy as np
from ultralytics import YOLO

# تحميل ملف الإعدادات
with open('D:/sign web/New folder/Sign-Language-Recognition/custom_data.yaml', 'r') as file:
    data = yaml.safe_load(file)

# تحميل النموذج
model = YOLO('D:/sign web/New folder/Sign-Language-Recognition/weights/best.pt')  # تأكد من أن هذا هو مسار النموذج المدرب الخاص بك

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# تعريف الألوان الممكنة للإطارات
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تطبيق النموذج على الإطار
    results = model(frame)

    # رسم الصناديق التنبؤية
    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()  # تحويل إلى numpy array
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = data['names'][int(cls)]
            
            # اختيار اللون من القائمة بشكل دائري
            color = colors[idx % len(colors)]
            
            # رسم المستطيل والنص على الإطار
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # تغيير السمك إلى 3
            cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # عرض الإطار
    cv2.imshow('Sign Language Detection', frame)

    # كسر الحلقة عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الكاميرا وإغلاق جميع النوافذ
cap.release()
cv2.destroyAllWindows()
