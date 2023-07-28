import cv2
from mtcnn.mtcnn import MTCNN


def detect_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces


def draw_faces(image, faces):
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return image


def main():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if ret:
            faces = detect_faces(frame)
            frame_with_faces = draw_faces(frame.copy(), faces)
            cv2.imshow('Video', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
