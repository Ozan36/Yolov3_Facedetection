import cv2   
import numpy as np  

# YOLO modelini yükleme
# YOLOv3-tiny konfigürasyon dosyasını ve ağırlıklarını yükleme
net = cv2.dnn.readNetFromDarknet("ObjectrecognitionYlv3Tny/yolov3-tiny.cfg", 
                                "ObjectrecognitionYlv3Tny/yolov3-tiny.weights")
# Ağın çıktı katmanlarının isimlerini al
output_layers = net.getUnconnectedOutLayersNames()

def yüz_tanima(image):
    """Görüntüde yüz tespiti yapar ve tespit edilen yüzlerin koordinatlarını döndürür"""
    # Görüntünün yüksekliğini ve genişliğini al
    h, w = image.shape[:2]
    
    # Görüntüyü YOLO modeline uygun formata dönüştürme (blob)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
    
    # Blobu modele girdi olarak verme
    net.setInput(blob)
    # İleri besleme yaparak çıktıları alma
    outputs = net.forward(output_layers)
    
    # Tespit edilen nesnelerin sınırlayıcı kutuları ve güven skorları
    boxes = []
    confidences = []
    
    # Tüm çıktı katmanlarını dolaş
    for output in outputs:
        # Her bir tespiti incele
        for detection in output:
            # En yüksek güven skorunu al
            confidence = detection[5:].max()
            # Eğer güven skoru eşik değerinden yüksekse
            if confidence > 0.4:
                # Sınırlayıcı kutu koordinatlarını orijinal görüntü boyutuna ölçekle
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                # Kutu koordinatlarını hesapla
                x, y = int(cx - bw/2), int(cy - bh/2)
                # Kutuyu ve güven skorunu listelere ekle
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
    
    # Çakışan kutuları ele (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # NMS sonrası kalan kutuları döndür (eğer varsa)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

def sekil_cizme(image, boxes):
    """Tespit edilen yüzlerin etrafına sınırlayıcı kutu çizer ve sayar"""
    # Her bir yüz için
    for x, y, w, h in boxes:
        # Yüzün etrafına dikdörtgen çiz
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Yüz etiketi ekle
        cv2.putText(image, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Tespit edilen yüz sayısını görüntüye yaz
    cv2.putText(image, f"Faces: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Kamera fonksiyonu
def camera():
    """Kameradan görüntü alır ve yüz tanıma işlemi yapar"""
    # Kamera bağlantısını başlat
    cap = cv2.VideoCapture(0)
    print("Kamera açıldı. Çıkmak için 'q'")
    
    # Sonsuz döngü (kullanıcı 'q' tuşuna basana kadar)
    while True:
        # Kameradan bir kare al
        ret, frame = cap.read()
        # Eğer kare alınamadıysa döngüyü kapat
        if not ret: break
        frame=cv2.flip(frame,1)
        # Yüz tespiti yap
        boxes = yüz_tanima(frame)
        # Tespit edilen yüzleri çiz
        sekil_cizme(frame, boxes)
        
        # Sonucu göster
        cv2.imshow("Yuz Tanima", frame)
        if cv2.waitKey(1) == ord('q'): break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()
camera()