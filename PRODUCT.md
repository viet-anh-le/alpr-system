# Product

## Register

product

## Users

Primary users are graduation-project reviewers, technical mentors, recruiters, and developers evaluating a Vietnamese ALPR pipeline. Secondary users are operators testing uploaded traffic footage or short marked events. They need to understand what the model saw, which plate text it inferred, how confident it was, and what evidence supports the result.

## Product Purpose

ALPR Vietnamese is an evidence-first web workbench for detecting, tracking, and recognizing Vietnamese license plates from video and live camera sources. Success means a reviewer can run a source through the pipeline, inspect recognized plates and rejected candidates, verify confidence and crops, and understand the thesis-grade computer vision system without reading backend code.

## Brand Personality

Precise, technical, trustworthy. The interface should feel like a forensic AI lab: calm, inspectable, data-rich, and honest about uncertainty.

## Anti-references

Avoid generic SaaS landing-page polish, decorative analytics dashboards, toy AI demos, glassy cyberpunk panels, and overloaded control-room visuals that hide the evidence trail.

## Design Principles

- Evidence first: pair every plate text with crop evidence, confidence, and track context.
- Progressive disclosure: keep model and preprocessing choices available without forcing them into the primary path.
- State legibility: idle, uploading, processing, done, failed, rejected, and saved states must be instantly distinguishable.
- Graduation honesty: claims, metrics, and states should feel documented and verifiable rather than salesy.
- Operator speed: every screen should present one clear next action.

## Accessibility & Inclusion

Target WCAG AA contrast for text and controls. Support keyboard navigation, visible focus rings, reduced motion, readable Vietnamese copy, and non-color-only status indicators.
