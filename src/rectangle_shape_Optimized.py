import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe_tool.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    edges = cv2.Canny(blur, 100, 200) # Higher thresholds = less noise

    # Increased threshold and minLineLength to filter noise
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = (x2 - x1), (y2 - y1)
            length = np.hypot(dx, dy)
            
            dx /= length
            dy /= length
            nx, ny = -dy, dx # Normal vector
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Probing for parallel edge
            max_dist = int(min(width, height) * 0.1)
            pos_edge = 0
            neg_edge = 0

            # Scan with direction-agnostic persistence
            for i in range(5, max_dist): # Start at 5 to skip the line itself
                px, py = int(cx + i * nx), int(cy + i * ny)
                if 0 <= px < width and 0 <= py < height:
                    if edges[py, px] == 255:
                        pos_edge = i
                        break

            for i in range(5, max_dist):
                mx, my = int(cx - i * nx), int(cy - i * ny)
                if 0 <= mx < width and 0 <= my < height:
                    if edges[my, mx] == 255:
                        neg_edge = i
                        break

            # VALIDATION: Only draw if we found a likely parallel edge (a rectangle)
            if pos_edge > 0 or neg_edge > 0:
                thickness = (pos_edge if pos_edge > 0 else 0) + (neg_edge if neg_edge > 0 else 0)
                if thickness < 15: continue # Ignore too-thin noise

                b = thickness / 2
                p1, p2, n = np.array([x1, y1]), np.array([x2, y2]), np.array([nx, ny])
                
                # Center the rectangle on the detected mass
                shift = (pos_edge - neg_edge) / 2
                center_p1 = p1 + shift * n
                center_p2 = p2 + shift * n

                pts = np.array([center_p1 + b*n, center_p2 + b*n, center_p2 - b*n, center_p1 - b*n], dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("Clean Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()