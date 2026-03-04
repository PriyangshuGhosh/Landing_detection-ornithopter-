import numpy as np
import cv2

cap = cv2.VideoCapture(0)
clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def get_profile_variance(img, x, y, nx, ny, dist=10):
    """Samples across the line and returns the standard deviation."""
    samples = []
    for i in range(-dist, dist):
        px, py = int(x + i * nx), int(y + i * ny)
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            samples.append(img[py, px])
    return np.std(samples) if len(samples) > 5 else 0

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe_tool.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges = cv2.Canny(blur, 60, 160)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 5 or abs(angle - 90) < 5: continue 

            length = np.hypot(x2 - x1, y2 - y1)
            nx, ny = -(y2 - y1)/length, (x2 - x1)/length

            v1 = get_profile_variance(enhanced, x1 + 0.25*(x2-x1), y1 + 0.25*(y2-y1), nx, ny)
            v2 = get_profile_variance(enhanced, (x1+x2)/2, (y1+y2)/2, nx, ny)
            v3 = get_profile_variance(enhanced, x1 + 0.75*(x2-x1), y1 + 0.75*(y2-y1), nx, ny)

            if min(v1, v2, v3) > 18: 
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Filtered Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
