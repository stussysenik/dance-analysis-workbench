## ADDED Requirements

### Requirement: Pipeline emits typed dancer track outputs
The system SHALL transform an input video segment into typed track records for up to two dancers, including frame spans, bounding boxes, optional masks, and confidence values.

#### Scenario: One dancer found
- **WHEN** analysis detects a single clear dancer in the selected segment
- **THEN** the output includes one typed track record with non-empty frame coverage and confidence scores

#### Scenario: Two dancers found
- **WHEN** analysis detects two dancers in the selected segment
- **THEN** the output includes two distinct track records with stable identities whenever confidence remains above the configured threshold

### Requirement: Pipeline degrades gracefully on uncertain frames
The system SHALL mark uncertain detections, occlusions, and track instabilities with explicit warning flags instead of failing the whole analysis run.

#### Scenario: Crossing dancers reduce confidence
- **WHEN** dancers overlap and the tracker cannot maintain identity confidently
- **THEN** the output marks the affected frames as unstable and preserves the rest of the result bundle

### Requirement: Pipeline supports pluggable visual analyzers
The system SHALL expose adapter interfaces for the shared visual backbone, track proposal logic, pose estimation, and optional part segmentation so model-backed and fallback analyzers can be swapped without changing the application contract.

#### Scenario: Fallback analyzer is active
- **WHEN** heavyweight model weights are unavailable
- **THEN** the pipeline still runs using configured fallback analyzers and produces schema-valid outputs
