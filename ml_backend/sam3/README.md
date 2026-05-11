# SAM3 Label Studio ML Backend

## Supported interactive inputs
- TextArea
- KeyPointLabels
- RectangleLabels

## Prompt priority
RectangleLabels > KeyPointLabels > TextArea

## Labeling config example
```xml
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="brush" toName="image">
    <Label value="defect" background="#FFA39E"/>
  </BrushLabels>
  <KeyPointLabels name="kp" toName="image" smart="true">
    <Label value="defect" background="#250DD3"/>
  </KeyPointLabels>
  <RectangleLabels name="rect" toName="image" smart="true">
    <Label value="defect" background="#FFC069"/>
  </RectangleLabels>
  <TextArea name="prompt" toName="image" perRegion="false"/>
</View>
```

## Run
- Docker
  - `cd label_studio_ml/sam3 && docker compose up --build`
- Local
  - `cd label_studio_ml/sam3 && pip install -r requirements-base.txt -r requirements.txt`
  - `../app/start.sh`

## Test
`pytest -q`
