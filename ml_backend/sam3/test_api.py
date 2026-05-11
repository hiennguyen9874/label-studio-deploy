import json

import numpy as np
import pytest
from model import NewModel
from tests.fixtures import make_task_payload, make_context_payload


@pytest.fixture
def client():
    from _wsgi import init_app

    app = init_app(model_class=NewModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_fixture_generates_box_context():
    ctx = make_context_payload(mode="box")
    assert ctx["result"][0]["type"] == "rectanglelabels"


def test_setup_sets_model_version_without_property_assignment_error():
    model = NewModel()
    assert model.get('model_version') == 'NewModel-v0.0.1'


def test_health_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200


def test_parse_textarea_prompt_payload():
    model = NewModel()
    context = {"result": [{"type": "textarea", "value": {"text": ["ear"]}}]}
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["text"] == "ear"


def test_parse_keypoints_payload():
    model = NewModel()
    context = {
        "result": [{
            "type": "keypointlabels",
            "value": {"x": 10, "y": 12, "keypointlabels": ["defect"]},
            "is_positive": 1,
        }]
    }
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["points"] == [[100, 120]]
    assert parsed["point_labels"] == [1]


def test_parse_rectangle_payload():
    model = NewModel()
    context = {
        "result": [{
            "type": "rectanglelabels",
            "value": {"x": 10, "y": 12, "width": 20, "height": 16, "rectanglelabels": ["defect"]},
        }]
    }
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["box_xyxy"] == [100, 120, 300, 280]


def test_priority_uses_box_over_keypoint_and_text():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [[10, 20]],
        "point_labels": [1],
        "box_xyxy": [10, 20, 30, 40],
    }
    assert model._select_prompt_mode(parsed) == "box"


def test_priority_uses_keypoint_over_text_when_no_box():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [[10, 20]],
        "point_labels": [1],
        "box_xyxy": None,
    }
    assert model._select_prompt_mode(parsed) == "keypoint"


def test_priority_uses_text_when_only_text_present():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [],
        "point_labels": [],
        "box_xyxy": None,
    }
    assert model._select_prompt_mode(parsed) == "text"


def _label_config():
    return "<View><Image name='image' value='$image'/><BrushLabels name='brush' toName='image'><Label value='defect'/></BrushLabels><RectangleLabels name='rect' toName='image' smart='true'><Label value='defect'/></RectangleLabels><KeyPointLabels name='kp' toName='image' smart='true'><Label value='defect'/></KeyPointLabels><TextArea name='txt' toName='image' smart='true'/></View>"


def _request_payload(mode="text"):
    return {
        "project": "1",
        "tasks": make_task_payload()["tasks"],
        "label_config": _label_config(),
        "params": {"context": make_context_payload(mode=mode)},
    }


def _assert_brush_response(payload):
    assert "results" in payload
    pred = payload["results"][0]
    assert pred["model_version"].startswith("NewModel")
    result = pred["result"][0]
    assert result["type"] == "brushlabels"
    assert result["value"]["format"] == "rle"
    assert "rle" in result["value"]


def test_predict_box_mode(client, monkeypatch):
    monkeypatch.setattr(NewModel, 'get_local_path', lambda self, url, task_id=None: '/tmp/fake.jpg')
    monkeypatch.setattr('model.Image.open', lambda path: ImageStub())
    monkeypatch.setattr(NewModel, '_run_sam3', lambda self, image, parsed: (np.ones((1000, 1000), dtype=np.uint8), 0.9))

    client.post('/setup', data=json.dumps({"project": "1", "schema": _label_config()}), content_type='application/json')
    response = client.post('/predict', data=json.dumps(_request_payload(mode="box")), content_type='application/json')

    assert response.status_code == 200
    _assert_brush_response(json.loads(response.data))


def test_predict_keypoint_mode(client, monkeypatch):
    monkeypatch.setattr(NewModel, 'get_local_path', lambda self, url, task_id=None: '/tmp/fake.jpg')
    monkeypatch.setattr('model.Image.open', lambda path: ImageStub())
    monkeypatch.setattr(NewModel, '_run_sam3', lambda self, image, parsed: (np.ones((1000, 1000), dtype=np.uint8), 0.8))

    client.post('/setup', data=json.dumps({"project": "1", "schema": _label_config()}), content_type='application/json')
    response = client.post('/predict', data=json.dumps(_request_payload(mode="keypoint")), content_type='application/json')

    assert response.status_code == 200
    _assert_brush_response(json.loads(response.data))


def test_predict_text_mode(client, monkeypatch):
    monkeypatch.setattr(NewModel, 'get_local_path', lambda self, url, task_id=None: '/tmp/fake.jpg')
    monkeypatch.setattr('model.Image.open', lambda path: ImageStub())
    monkeypatch.setattr(NewModel, '_run_sam3', lambda self, image, parsed: (np.ones((1000, 1000), dtype=np.uint8), 0.7))

    client.post('/setup', data=json.dumps({"project": "1", "schema": _label_config()}), content_type='application/json')
    response = client.post('/predict', data=json.dumps(_request_payload(mode="text")), content_type='application/json')

    assert response.status_code == 200
    _assert_brush_response(json.loads(response.data))


def test_predict_mixed_prompts_uses_box_priority(client, monkeypatch):
    selected_mode = {"value": None}

    def fake_run_sam3(self, image, parsed):
        selected_mode["value"] = self._select_prompt_mode(parsed)
        return np.ones((1000, 1000), dtype=np.uint8), 0.9

    monkeypatch.setattr(NewModel, 'get_local_path', lambda self, url, task_id=None: '/tmp/fake.jpg')
    monkeypatch.setattr('model.Image.open', lambda path: ImageStub())
    monkeypatch.setattr(NewModel, '_run_sam3', fake_run_sam3)

    mixed = make_context_payload(mode="text")["result"] + make_context_payload(mode="keypoint")["result"] + make_context_payload(mode="box")["result"]
    payload = {
        "project": "1",
        "tasks": make_task_payload()["tasks"],
        "label_config": _label_config(),
        "params": {"context": {"result": mixed}},
    }

    client.post('/setup', data=json.dumps({"project": "1", "schema": _label_config()}), content_type='application/json')
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')

    assert response.status_code == 200
    _assert_brush_response(json.loads(response.data))
    assert selected_mode["value"] == "box"


def test_predict_returns_empty_when_context_missing(client):
    label_config = "<View><Image name='image' value='$image'/><BrushLabels name='brush' toName='image'><Label value='defect'/></BrushLabels></View>"
    client.post('/setup', data=json.dumps({"project": "1", "schema": label_config}), content_type='application/json')

    request = {
        "project": "1",
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "label_config": label_config,
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    payload = json.loads(response.data)
    assert payload["results"][0]["result"] == []


class ImageStub:
    def convert(self, mode):
        return np.zeros((1000, 1000, 3), dtype=np.uint8)
