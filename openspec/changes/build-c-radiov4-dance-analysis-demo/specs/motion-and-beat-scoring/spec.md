## ADDED Requirements

### Requirement: System computes frame-level COM and motion qualities
The system SHALL compute per-frame biomechanics records for each tracked dancer, including a 2D center-of-mass proxy, velocity, acceleration, travel distance, and expansion proxy, together with confidence values.

#### Scenario: Frame-level metrics generated
- **WHEN** a dancer track contains pose data for the selected segment
- **THEN** the system emits a biomechanics record for each covered frame

### Requirement: System computes basic beat alignment for the selected segment
The system SHALL estimate a beat grid for the selected segment and compute a basic on-beat score from dancer motion events, including confidence and warning fields.

#### Scenario: Beat score generated
- **WHEN** the selected segment contains analyzable audio
- **THEN** the system emits beat timestamps, an estimated tempo, and a segment-level on-beat score

### Requirement: Visualization surfaces uncertainty in computed metrics
The system SHALL surface low-confidence COM, pose, and beat outputs in both structured results and the annotated preview.

#### Scenario: Low-confidence metrics displayed
- **WHEN** the confidence for a metric falls below the configured threshold
- **THEN** the preview and structured outputs include a warning that the metric is approximate
