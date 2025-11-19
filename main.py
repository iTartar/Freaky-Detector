import cv2
import mediapipe as mp
import numpy as np
import imageio
import time
from pathlib import Path

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# GIF paths
tongue_gif = "assets/tongue.gif"
closed_eyes_gif = "assets/closed_eyes.gif"
thinking_gif = "assets/thinking.gif"
pointing_gif = "assets/pointing.gif"  # Add pointing gesture GIF

# Load GIFs
def load_gif(path):
    try:
        gif_frames = imageio.mimread(path)
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif_frames]
        return frames_bgr
    except Exception as e:
        print(f"Error loading GIF {path}: {e}")
        return None

tongue_frames = load_gif(tongue_gif)
eyes_frames = load_gif(closed_eyes_gif)
thinking_frames = load_gif(thinking_gif)
pointing_frames = load_gif(pointing_gif)

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    ear = (v1 + v2) / (2.0 * h)
    return ear

def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

def detect_thinking_gesture(hand_landmarks, face_landmarks, frame_shape):
    """Detect if hand is near chin/cheek in thinking pose"""
    h, w, _ = frame_shape
    
    # Get chin position (landmark 152)
    chin = face_landmarks[152]
    chin_pos = np.array([chin.x, chin.y])
    
    # Get cheek positions
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    left_cheek_pos = np.array([left_cheek.x, left_cheek.y])
    right_cheek_pos = np.array([right_cheek.x, right_cheek.y])
    
    # Get lower face area (jaw)
    jaw_left = face_landmarks[172]
    jaw_right = face_landmarks[397]
    jaw_left_pos = np.array([jaw_left.x, jaw_left.y])
    jaw_right_pos = np.array([jaw_right.x, jaw_right.y])
    
    # Get hand landmarks
    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]
    
    wrist_pos = np.array([wrist.x, wrist.y])
    index_pos = np.array([index_tip.x, index_tip.y])
    middle_pos = np.array([middle_tip.x, middle_tip.y])
    thumb_pos = np.array([thumb_tip.x, thumb_tip.y])
    index_mcp_pos = np.array([index_mcp.x, index_mcp.y])
    
    # Calculate distances to face key points
    distances = [
        np.linalg.norm(wrist_pos - chin_pos),
        np.linalg.norm(index_pos - chin_pos),
        np.linalg.norm(middle_pos - chin_pos),
        np.linalg.norm(thumb_pos - chin_pos),
        np.linalg.norm(index_mcp_pos - chin_pos),
        np.linalg.norm(wrist_pos - left_cheek_pos),
        np.linalg.norm(wrist_pos - right_cheek_pos),
        np.linalg.norm(index_pos - jaw_left_pos),
        np.linalg.norm(index_pos - jaw_right_pos),
    ]
    
    # Check if any hand part is close to face (threshold 0.12)
    min_distance = min(distances)
    return min_distance < 0.12

def detect_pointing_up(hand_landmarks):
    """Detect pointing up gesture - index finger extended upward"""
    
    # Get hand landmarks
    wrist = hand_landmarks.landmark[0]
    
    # Index finger
    index_mcp = hand_landmarks.landmark[5]  # Base of index
    index_pip = hand_landmarks.landmark[6]  # Second joint
    index_dip = hand_landmarks.landmark[7]  # Third joint
    index_tip = hand_landmarks.landmark[8]  # Tip
    
    # Other fingers (should be folded)
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    
    # Convert to numpy arrays
    wrist_pos = np.array([wrist.x, wrist.y])
    index_mcp_pos = np.array([index_mcp.x, index_mcp.y])
    index_pip_pos = np.array([index_pip.x, index_pip.y])
    index_dip_pos = np.array([index_dip.x, index_dip.y])
    index_tip_pos = np.array([index_tip.x, index_tip.y])
    
    middle_tip_pos = np.array([middle_tip.x, middle_tip.y])
    ring_tip_pos = np.array([ring_tip.x, ring_tip.y])
    pinky_tip_pos = np.array([pinky_tip.x, pinky_tip.y])
    thumb_tip_pos = np.array([thumb_tip.x, thumb_tip.y])
    
    # Check 1: Index finger is pointing up (tip is above base)
    index_pointing_up = index_tip.y < index_mcp.y - 0.08
    
    # Check 2: Index finger is extended (joints are in line vertically)
    index_extended = (index_pip.y < index_mcp.y and 
                     index_dip.y < index_pip.y and 
                     index_tip.y < index_dip.y)
    
    # Check 3: Other fingers are curled (tips below or near their base)
    middle_curled = middle_tip.y > index_mcp.y - 0.05
    ring_curled = ring_tip.y > index_mcp.y - 0.05
    pinky_curled = pinky_tip.y > index_mcp.y - 0.05
    
    # Check 4: Index finger is relatively straight (horizontal distance is small)
    index_horizontal_offset = abs(index_tip.x - index_mcp.x)
    index_straight = index_horizontal_offset < 0.06
    
    # All conditions must be met
    pointing = (index_pointing_up and index_extended and 
                middle_curled and ring_curled and pinky_curled and
                index_straight)
    
    return pointing

# Thresholds
EYE_AR_THRESH = 0.20
MOUTH_AR_THRESH = 0.55

cap = cv2.VideoCapture(0)
frames_for_gif = []

reaction_mode = None
reaction_index = 0

print("🎥 Freak Detector Started!")
print("👀 Close your eyes -> 'hell nah'")
print("👅 Stick tongue out -> 'freak of nature'")
print("🤔 Hand on chin -> 'pondering freak'")
print("☝️ Point finger up -> 'genius freak'")
print("Press Q to quit...")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process face
    face_results = face_mesh.process(rgb)
    
    # Process hands
    hand_results = hands.process(rgb)
    
    thinking_detected = False
    pointing_detected = False
    eyes_closed = False
    tongue_out = False
    
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        # Eye detection
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [263, 387, 385, 362, 380, 373]
        left_EAR = eye_aspect_ratio(landmarks, left_eye_idx)
        right_EAR = eye_aspect_ratio(landmarks, right_eye_idx)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        
        # Mouth detection
        mar = mouth_aspect_ratio(landmarks)

        eyes_closed = avg_EAR < EYE_AR_THRESH
        tongue_out = mar > MOUTH_AR_THRESH
        
        # Hand gesture detection
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check for pointing up gesture
                if detect_pointing_up(hand_landmarks):
                    pointing_detected = True
                # Check for thinking gesture
                elif detect_thinking_gesture(hand_landmarks, landmarks, frame.shape):
                    thinking_detected = True
        
        # Priority: pointing > thinking > eyes closed > tongue out
        if pointing_detected:
            reaction_mode = "pointing"
            cv2.putText(frame, "genius freak", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif thinking_detected:
            reaction_mode = "thinking"
            cv2.putText(frame, "pondering freak", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        elif eyes_closed:
            reaction_mode = "eyes"
            cv2.putText(frame, "hell nah", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif tongue_out:
            reaction_mode = "tongue"
            cv2.putText(frame, "freak of nature", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            reaction_mode = None
            cv2.putText(frame, "Normal", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display main camera feed
    cv2.imshow("Freak Detector", frame)

    # Reaction window
    if reaction_mode == "pointing" and pointing_frames:
        gif_frame = pointing_frames[reaction_index % len(pointing_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "thinking" and thinking_frames:
        gif_frame = thinking_frames[reaction_index % len(thinking_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "eyes" and eyes_frames:
        gif_frame = eyes_frames[reaction_index % len(eyes_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "tongue" and tongue_frames:
        gif_frame = tongue_frames[reaction_index % len(tongue_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    else:
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(blank, "Not Freaky", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("Reaction", blank)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()