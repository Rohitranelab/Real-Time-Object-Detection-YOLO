from collections import Counter

def count_objects(detections):
    try:
        if not detections:
            return {}

        object_names = [detection["class"] for detection in detections]
        return dict(Counter(object_names))

    except Exception as e:
        print(e)

def total_objects(detections):
    try:
        return len(detections)

    except Exception as e:
        print(e)

def average_confidence(detections):
    try:
        if not detections:
            return 0

        total = sum(detection["confidence"] for detection in detections)
        return total / len(detections)

    except Exception as e:
        print(e)