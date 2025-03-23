import cv2
from deepface import DeepFace
import time

print("Starting webcam access...")

reference_img = cv2.imread('reference.jpg')

if reference_img is None:
    print("Error: reference.jpg not found.")
    exit()

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam opened successfully.")

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    print(f"Captured frame {counter}")

    if counter % 30 == 0:
        print("Verifying face...") 
        try:
            result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
            print(f"Verification result: {result}") 
            face_match = result['verified']

            analysis = DeepFace.analyze(frame, actions=['emotion', 'age'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            age = analysis[0]['age']
            print(f"Detected Emotion: {emotion}")
            print(f"Detected Age: {age}")
        except Exception as e:
            print(f"Error during verification: {e}")
            face_match = False
            emotion = "N/A"
            age = "N/A"

    counter += 1

    if face_match:
        cv2.putText(frame, 'MATCH!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'NO MATCH!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.putText(frame, f"Emotion: {emotion}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Age: {age}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow('Webcam', frame)

    
    time.sleep(0.03)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
