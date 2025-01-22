import cv2 as cv
face_detect = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
image_count = 0
total_people_detected = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    person_count = len(faces)
    total_people_detected = max(total_people_detected, person_count)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        text = f"Person {person_count}"
        cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.1, (120, 220, 255), 3)

    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv.waitKey(1) & 0xFF == ord('s'):
        image_name = f"captured_image_{image_count}.jpg"
        cv.imwrite(image_name, frame)
        image_count += 1

print("No of people detected :- ", total_people_detected)
cap.release()
cv.destroyAllWindows()
