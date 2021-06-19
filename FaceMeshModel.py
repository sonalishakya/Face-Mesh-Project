import cv2 as cv
import mediapipe as mp
import time


class faceMeshDetector():
    def __init__(self, mode=False, maxFaces=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.mode, self.maxFaces, self.detectionCon, self.trackCon)
        self.drawSpc = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, frame, draw=True):
        self.frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(self.frame_RGB)
        all_faces = []
        if self.result.multi_face_landmarks:
            for faces in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, faces, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpc, self.drawSpc)
                face = []
                for id, lm in enumerate(faces.landmark):
                    # print(id, lm)
                    h, w, c = frame.shape
                    x, y, z = int(lm.x*w), int(lm.y*h), int(lm.z*c)
                    print(id, x, y, z)
                    face.append([x, y])
                all_faces.append(face)
        return frame, all_faces


def main():
    camera = cv.VideoCapture(0)
    pTime = 0
    detector = faceMeshDetector()
    while True:
        isTrue, frame = camera.read()
        frame, faces = detector.findFaceMesh(frame)
        if len(faces) != 0:
            print(len(faces))
        # adding frames per second
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame, f"FPS {int(fps)}", (20, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow('Camera', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
