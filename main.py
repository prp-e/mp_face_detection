import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

while camera.isOpened():
    _, frame = camera.read()

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        try:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame)
            for face in results.detections:
                x_min = face.location_data.relative_bounding_box.xmin
                y_min = face.location_data.relative_bounding_box.ymin

                x_max = face.location_data.relative_bounding_box.xmin + face.location_data.relative_bounding_box.width
                y_max = face.location_data.relative_bounding_box.ymin + face.location_data.relative_bounding_box.height

                rect_start = mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, frame.shape[1], frame.shape[0])
                rect_end = mp_drawing._normalized_to_pixel_coordinates(x_max, y_max, frame.shape[1], frame.shape[0])
                
                cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)
        except:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Cam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
