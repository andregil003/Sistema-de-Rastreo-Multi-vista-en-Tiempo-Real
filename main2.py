import cv2
from camera_stream import load_cameras_from_config

def main():
    cameras = load_cameras_from_config("config.json")

    while True:
        frames = []
        for cam in cameras:
            frame = cam.read()
            if frame is None:
                frame = cv2.putText(
                    img = cv2.imread("black.jpg") if cv2.haveImageReader("black.jpg") else
                    cv2.cvtColor(cv2.UMat(240, 320, cv2.CV_8UC3), cv2.COLOR_BGR2RGB).get(),
                    text = f"No Frame {cam.cam_id}", org = (50, 50),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,
                    color = (0, 0, 255), thickness = 2
                )
            frames.append(frame)

        if len(frames) == 0:
            break

        # Resize and stack for display
        resized = [cv2.resize(f, (320, 240)) for f in frames]
        combined = cv2.hconcat(resized)

        cv2.imshow("Multi-Cam View", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cam in cameras:
        cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
