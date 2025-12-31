import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Delay between rectangle detections (seconds)
detection_interval = 0.2
last_detection_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = clahe_tool.apply(gray)
    blur = cv2.GaussianBlur(clahe, (7, 7), 3)  # smoother and more uniform
    edges = cv2.Canny(blur, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Only run rectangle detection every detection_interval seconds
    if time.time() - last_detection_time > detection_interval:
        last_detection_time = time.time()
        lines = cv2.HoughLinesP(
            edges_dilated,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=80,  # filter tiny edges
            maxLineGap=10
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length < 80:  # ignore short lines
                    continue

                dx /= length
                dy /= length

                # Perpendicular unit vector
                nx = -dy
                ny = dx

                # Line midpoint
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Perpendicular thickness detection
                max_dist = int(min(width, height) * 0.08)
                pos_edge = 0
                neg_edge = 0

                # Scan forward
                for i in range(1, max_dist):
                    px = int(cx + i * nx)
                    py = int(cy + i * ny)
                    if 0 <= px < width and 0 <= py < height:
                        if edges_dilated[py, px] == 255:
                            # Check if edge persists for 3 pixels
                            if all(edges_dilated[min(height-1, py+j), px] == 255 for j in range(3)):
                                pos_edge = i
                                break

                # Scan backward
                for i in range(1, max_dist):
                    mx = int(cx - i * nx)
                    my = int(cy - i * ny)
                    if 0 <= mx < width and 0 <= my < height:
                        if edges_dilated[my, mx] == 255:
                            if all(edges_dilated[min(height-1, my+j), mx] == 255 for j in range(3)):
                                neg_edge = i
                                break

                thickness = pos_edge + neg_edge
                thickness = max(thickness, 20)

                # Rectangle corners
                b = thickness / 2
                p1 = np.array([x1, y1], dtype=np.float32)
                p2 = np.array([x2, y2], dtype=np.float32)
                n = np.array([nx, ny], dtype=np.float32)

                A = p1 + b * n
                B = p2 + b * n
                C = p2 - b * n
                D = p1 - b * n

                corners = np.array([A, B, C, D], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Rectangular Structure Detection", frame)
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
