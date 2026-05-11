def make_task_payload(image_url="http://localhost/image.jpg"):
    return {"tasks": [{"id": 1, "data": {"image": image_url}}]}


def make_context_payload(mode="text"):
    if mode == "box":
        return {
            "result": [{
                "type": "rectanglelabels",
                "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["defect"]},
                "original_width": 1000,
                "original_height": 1000,
            }]
        }
    if mode == "keypoint":
        return {
            "result": [{
                "type": "keypointlabels",
                "value": {"x": 10, "y": 10, "keypointlabels": ["defect"]},
                "is_positive": 1,
                "original_width": 1000,
                "original_height": 1000,
            }]
        }
    return {
        "result": [{
            "type": "textarea",
            "value": {"text": ["ear"]},
            "original_width": 1000,
            "original_height": 1000,
        }]
    }
