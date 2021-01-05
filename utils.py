def get_biggest_keypoint(keypoint) -> float:
    biggest: float = 0
    for kp in keypoint:
        if kp.size > biggest:
            biggest = kp.size


    return biggest
