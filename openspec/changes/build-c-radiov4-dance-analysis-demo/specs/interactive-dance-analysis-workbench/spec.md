## ADDED Requirements

### Requirement: User can analyze a local dance video interactively
The system SHALL provide a local interactive workbench that accepts a local video file, allows the user to select a time segment, and runs analysis for one or two dancers.

#### Scenario: Analyze selected segment
- **WHEN** a user provides a local video path and selects a start and end time
- **THEN** the system runs analysis only for that segment and produces a reviewable result bundle

### Requirement: User can inspect overlays without visual collisions
The system SHALL render an annotated preview with independent toggles for track boxes, masks, skeletons, COM traces, and beat markers, and it MUST support primary and secondary dancer focus modes so overlays remain legible.

#### Scenario: Toggle overlay layers
- **WHEN** the user enables or disables individual overlay layers
- **THEN** the preview updates to show only the selected layers while preserving dancer identity labels

### Requirement: User can correct unstable dancer identity assignments
The system SHALL expose a human-in-the-loop refinement flow that allows a user to re-run analysis for a segment after providing corrected dancer seeds or focus selection.

#### Scenario: Re-run after correction
- **WHEN** the system marks a track as unstable and the user submits corrected identity hints
- **THEN** the system reprocesses the segment and returns an updated result bundle
