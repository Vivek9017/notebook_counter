from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Camera not detected")

line_position = int(frame.shape[0] * 0.5)

count = 0
counted_ids = set()
previous_positions = {}

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.2)

    for r in results:
        boxes = r.boxes

        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            track_id = int(box.id[0])
            class_name = model.names[cls]

            if class_name in ["book"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_y = y2

                # 🔥 SIMPLIFIED & FIXED COUNTING LOGIC
                if bottom_y > line_position and track_id not in counted_ids:
                    count += 1
                    counted_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{class_name} ID:{track_id}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255,0,0), 2)

    cv2.line(frame, (0,line_position),
             (frame.shape[1], line_position),
             (0,0,255), 3)

    cv2.putText(frame, f"Notebook Count: {count}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,255,0),
                3)

    cv2.imshow("Live Notebook Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()