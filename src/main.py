from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import os
import cv2

FILE_PATH = os.path.dirname(os.path.abspath(__file__)) # File is in src/
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_PATH,"..")) # Now it goes back to project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "models","hand_landmarker.task") # It shold be able to find the model here
IMAGE_PATH = os.path.join(PROJECT_ROOT, "samples")

CAMERA_INDEX = 0

MAX_DISPLAY_WIDTH = 1280  # shrink big photos so the window fits the screen

# Pairs of landmark indices to join with lines (standard 21-point hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]

def process_image(image):
    frame = cv2.imread(image)

    if frame is None:
        raise FileNotFoundError(f"Image not found at {image}")

    with create_landmarker(vision.RunningMode.IMAGE) as landmarker:
        results = landmarker.detect(to_mp_image(frame))

    annotated_image = frame.copy()

    draw_hands(annotated_image, results)

    cv2.imshow("Hand Landmarks", annotated_image)
    cv2.imwrite("result-"+os.path.splitext(os.path.basename(image))[0]+".jpg", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_stream():
    pass

def create_landmarker(mode, result_callback=None):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path= MODEL_PATH),
        running_mode = mode,
        num_hands = 3,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback = result_callback,
    )
    return vision.HandLandmarker.create_from_options(options)

def to_mp_image(bgr_frame):
    # Mediapipe needs a RGB image but opencv imread frame is BGR coded
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_hands(img, result):

    if result is None:
        print("No results to show!!")
        return

    h, w = img.shape[:2]
    thickness = max(2, w // 500)

    """
    hand is a list of 21 landmark objects for one detected hand, index 0 = wrist, 4 = thumb tip, 8 = index fingertip, and so on.
    lm is one landmark. It has .x, .y and .z.
    lm.x and lm.y are normalized: values from 0 to 1 as a fraction of the image size. x = 0 is the left edge and x = 1 the right edge; y = 0 is the top and y = 1 the bottom.
    w and h are the image width and height in pixels, from frame.shape[:2].
    lm.x * w turns the fraction into a pixel position, and int() rounds it down to a whole pixel, since OpenCV needs integer coordinates.
    ( … , … ) makes an (x, y) tuple.
    The result pts is a list of 21 (x, y) tuples, in the same order as the landmarks.
    """

    for i, hand in enumerate(result.hand_landmarks):
        # Landmarks are normalised 0-1, so scale them to pixel coordinates
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), thickness)
        for p in pts:
            cv2.circle(img, p, thickness*2, (0, 0, 255), -1)

        label = result.handedness[i][0].category_name
        score = result.handedness[i][0].score

        wrist_x, wrist_y = pts[0]
        cv2.putText(img, f"{label} {score:.2f}", (wrist_x - 40, wrist_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness * 0.4, (0, 0, 0),
                    thickness)

def find_pictures(image_path):

    if not os.path.isdir(image_path):
        raise NotADirectoryError(f"Not a folder: {image_path}")

    extension = (".jpg", ".jpeg")
    images = []

    for name in os.listdir(image_path):
        path = os.path.join(image_path, name)
        if os.path.isfile(path) and name.lower().endswith(extension):
            images.append(path)

    return images

if __name__ == "__main__":

    for path in find_pictures(IMAGE_PATH):
        process_image(path)


