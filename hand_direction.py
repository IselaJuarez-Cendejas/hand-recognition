import numpy as np
import cv2


TEXT_COLOR = (243,236,27)
BAR_COLOR = (51,255,51)
LINE_COLOR = (255,255,255)
LM_COLOR = (255,51,255)
LABEL_COLOR = (102,51,0)

THUMB_STATES = {
    0: ['straight', (121,49,255)],
    1: ['bent', (243,166,56)],
    2: ['closed', (107,29,92)]
}
NON_THUMB_STATES = {
    0: ['straight', (121,49,255)],
    1: ['claw', (76,166,255)],
    2: ['bent', (243,166,56)],
    3: ['closed', (178,30,180)],
    4: ['clenched', (107,29,92)]
}


def find_boundary_landmarks(landmarks):
    """ Get the landmarks with maximum x, minimum x, maximum y, and minimum y values. """
    x_values = landmarks[:, 0]
    y_values = landmarks[:, 1]
    max_x_lm, min_x_lm = np.argmax(x_values), np.argmin(x_values)
    max_y_lm, min_y_lm = np.argmax(y_values), np.argmin(y_values)

    return [max_x_lm, min_x_lm, max_y_lm, min_y_lm]


def determine_hand_direction(landmarks, hand_label):
    """ Determine the hand's direction and facing. """
    direction = None
    facing = None
    knuckle_joints = [5, 9, 13, 17]
    wrist_lm = landmarks[0]
    thumb_mcp_lm = landmarks[1]
    pinky_mcp_lm = landmarks[17]

    avg_knuckle_x = np.mean(landmarks[knuckle_joints, 0])
    avg_knuckle_y = np.mean(landmarks[knuckle_joints, 1])

    knuckle_wrist_x_diff = np.absolute(avg_knuckle_x - wrist_lm[0])
    knuckle_wrist_y_diff = np.absolute(avg_knuckle_y - wrist_lm[1])

    if knuckle_wrist_x_diff > knuckle_wrist_y_diff:
        if avg_knuckle_x < wrist_lm[0]:
            direction = 'left'
            if hand_label == 'left':
                facing = 'front' if thumb_mcp_lm[1] < pinky_mcp_lm[1] else 'back'
            else:
                facing = 'front' if thumb_mcp_lm[1] > pinky_mcp_lm[1] else 'back'
        else:
            direction = 'right'
            if hand_label == 'left':
                facing = 'front' if thumb_mcp_lm[1] > pinky_mcp_lm[1] else 'back'
            else:
                facing = 'front' if thumb_mcp_lm[1] < pinky_mcp_lm[1] else 'back'
    else:
        if avg_knuckle_y < wrist_lm[1]:
            direction = 'up'
            if hand_label == 'left':
                facing = 'front' if thumb_mcp_lm[0] > pinky_mcp_lm[0] else 'back'
            else:
                facing = 'front' if thumb_mcp_lm[0] < pinky_mcp_lm[0] else 'back'
        else:
            direction = 'down'
            if hand_label == 'left':
                facing = 'front' if thumb_mcp_lm[0] < pinky_mcp_lm[0] else 'back'
            else:
                facing = 'front' if thumb_mcp_lm[0] > pinky_mcp_lm[0] else 'back'

    return direction, facing


def calculate_landmark_distance(landmark1, landmark2, dimension=2):
    """ Calculate the distance between two landmarks. """
    vec = landmark2[:dimension] - landmark1[:dimension]
    distance = np.linalg.norm(vec)
    return distance


def calculate_angle_between_points(points):
    """ Calculate the angle between three points. """
    vec1 = points[0][:2] - points[1][:2]
    vec2 = points[2][:2] - points[1][:2]
    cross_product = np.cross(vec1, vec2)
    dot_product = np.dot(vec1, vec2)
    angle = np.absolute(np.arctan2(cross_product, dot_product))
    return angle


def calculate_thumb_angle(joint_positions, hand_label, hand_facing):
    """ Calculate the thumb angle between three points. """
    vec1 = joint_positions[0][:2] - joint_positions[1][:2]
    vec2 = joint_positions[2][:2] - joint_positions[1][:2]
    if hand_label == 'left':
        cross_product = np.cross(vec1, vec2) if hand_facing == 'front' else np.cross(vec2, vec1)
    else:
        cross_product = np.cross(vec2, vec1) if hand_facing == 'front' else np.cross(vec1, vec2)
    dot_product = np.dot(vec1, vec2)
    angle = np.arctan2(cross_product, dot_product)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def define_finger_state(joint_angles, threshold_values):
    """ Define a finger's state based on its joint angles. """
    accumulated_angle = joint_angles.sum()
    finger_state = None
    updated_thresholds = threshold_values.copy()
    updated_thresholds.append(-np.inf)
    updated_thresholds.insert(0, np.inf)
    for i in range(len(updated_thresholds) - 1):
        if updated_thresholds[i] > accumulated_angle >= updated_thresholds[i + 1]:
            finger_state = i
            break
    return finger_state


def map_detected_gesture(gesture_templates, finger_states, landmarks, wrist_angle, hand_direction, boundary_info):
    """ Map detected gesture features to a pre-defined gesture template. """
    detected_gesture = None
    distance = calculate_landmark_distance(landmarks[0], landmarks[5])
    threshold = distance / 4
    for gesture, template in gesture_templates.items():
        count = 0
        flag = 0
        for i in range(len(finger_states)):
            if finger_states[i] not in template['finger_states'][i]:
                flag = 1
                break
        if flag == 0:
            count += 1
        if template['wrist_angle'][0] < wrist_angle < template['wrist_angle'][1]:
            count += 1
        if template['direction'] == hand_direction:
            count += 1
        if template['overlap'] is None:
            count += 1
        else:
            flag = 0
            for lm1, lm2 in template['overlap']:
                if calculate_landmark_distance(landmarks[lm1], landmarks[lm2]) > threshold:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        if template['boundary'] is None:
            count += 1
        else:
            flag = 0
            for bound, lm in template['boundary'].items():
                if boundary_info[bound] not in lm:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        if count == 5:
            detected_gesture = gesture
            break
    return detected_gesture


def draw_transparent_rectangle(img, point1, point2, alpha=0.5, beta=0.5):
    """ Draw a transparent rectangle. """
    sub_image = img[point1[1]:point2[1], point1[0]:point2[0]]
    white_rectangle = np.ones(sub_image.shape, dtype=np.uint8) * 255
    result = cv2.addWeighted(sub_image, alpha, white_rectangle, beta, 1.0)
    img[point1[1]:point2[1], point1[0]:point2[0]] = result


def draw_fingertips(landmarks, finger_states, img):
    """ Draw fingertips based on finger states. """
    width = img.shape[1]
    radius = int(width / 100)
    for i in range(5):
        fingertip = landmarks[4 * (i + 1)]
        if i == 0:
            color = THUMB_STATES[finger_states[i]][1]
        else:
            color = NON_THUMB_STATES[finger_states[i]][1]
        cv2.circle(img, fingertip[:2], radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, fingertip[:2], radius + 1, (255, 255, 255), int(radius / 5), lineType=cv2.LINE_AA)


def draw_bounding_box(landmarks, detected_gesture, img):
    """ Draw a bounding box around the detected hand with a gesture label. """
    width = img.shape[1]
    thickness = int(width / 40)
    x_values = landmarks[:, 0]
    y_values = landmarks[:, 1]
    x_max, x_min = np.max(x_values), np.min(x_values)
    y_max, y_min = np.max(y_values), np.min(y_values)
    draw_transparent_rectangle(img, (x_min - thickness, y_min - thickness - 40), (x_max + thickness, y_min - thickness))
    cv2.rectangle(img, (x_min - thickness, y_min - thickness), (x_max + thickness, y_max + thickness), LINE_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{detected_gesture}', (x_min - thickness + 5, y_min - thickness - 10), 0, 1, LABEL_COLOR, 3, lineType=cv2.LINE_AA)


def display_hand_info(img, hand):
    """ Display hand information. """
    w = img.shape[1]
    tor = int(w / 40)

    landmarks = hand['landmarks']
    label = hand['label']
    wrist_angle = hand['wrist_angle']
    direction = hand['direction']
    facing = hand['facing']

    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)

    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor),
                  LINE_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'LABEL: {label} hand', (x_min-tor,y_min-4*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'DIRECTION: {direction}', (x_min-tor,y_min-3*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'FACING: {facing}', (x_min-tor,y_min-2*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'WRIST ANGLE: {round(wrist_angle,1)}', (x_min-tor,y_min-tor-10),
                0, 0.6, LINE_COLOR, 2, lineType=cv2.LINE_AA)


def draw_landmarks(img, pt1, pt2, color=LINE_COLOR):
    """ Draw two landmarks and the connection line. """
    cv2.circle(img, pt1, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt2, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.line(img, pt1, pt2, color, 3)
