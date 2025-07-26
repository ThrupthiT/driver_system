import cv2
import time
from twilio.rest import Client

# Twilio credentials (replace with your actual values)
ACCOUNT_SID = 'account_sid'
AUTH_TOKEN = 'auth_token'
TWILIO_PHONE_NUMBER = ''   # Your Twilio number
EMERGENCY_CONTACT_NUMBER = ''  # Recipient's number

# Initialize Twilio client
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Time threshold if face not detected (in seconds)
ALERT_THRESHOLD = 10
face_last_seen = time.time()
alert_triggered = False

# Emergency action function with SMS
def trigger_emergency():
    print("🚨 ALERT: Driver collapse suspected! Triggering emergency protocol.")
    try:
        message = twilio_client.messages.create(
            body="🚨 EMERGENCY ALERT: Possible driver collapse detected. Immediate attention required!",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER
        )
        print("✅ SMS sent. SID:", message.sid)
    except Exception as e:
        print("❌ Failed to send SMS:", e)

# Real-time monitoring loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error accessing webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        face_last_seen = time.time()
        alert_triggered = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Driver Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        time_since_last_seen = time.time() - face_last_seen

        if time_since_last_seen >= ALERT_THRESHOLD and not alert_triggered:
            trigger_emergency()
            alert_triggered = True

        cv2.putText(frame, f"No face detected for {int(time_since_last_seen)}s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Driver Health Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
