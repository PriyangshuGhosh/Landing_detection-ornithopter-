import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=10000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.random.normal(mean, std, gray.shape)
    noisy = gray.astype(np.uint8) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Press SPACE to capture image | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1)

    if key == 32:  
        image = frame.copy()
        break
    elif key == 27: 
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

noisy_image = add_gaussian_noise(image, std=15)

cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Noisy Image", noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
