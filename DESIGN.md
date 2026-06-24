# Design

## Theme

The default surface is a restrained dark forensic lab: deep neutral background, crisp panel boundaries, cyan/blue focus accents, green success, amber uncertainty, and red failure. Landing and auth may use larger storytelling sections, but they must inherit the same technical identity.

## Typography

- UI font: IBM Plex Sans, then system sans-serif fallback.
- Data and plate font: JetBrains Mono.
- Use fixed product UI sizes. Keep labels readable; avoid 8-9px UI text except tiny metadata inside evidence thumbnails.
- Use tabular numerals for confidence, frame counts, and timestamps.

## Color Tokens

Use OKLCH CSS variables defined in `web/src/index.css`:

- `--color-bg`, `--color-bg-elevated`, `--color-surface`, `--color-panel`
- `--color-border`, `--color-border-strong`
- `--color-text`, `--color-text-muted`, `--color-text-subtle`
- `--color-accent`, `--color-accent-strong`
- `--color-success`, `--color-warning`, `--color-danger`, `--color-info`

Accent color is for primary actions, active selections, and model state. It is not decorative filler.

## Components

Shared primitives live under `web/src/components/ui/`: buttons, icon buttons, segmented controls, fields, selects, badges, progress, toast, drawer/dialog, empty states, and skeletons. Feature surfaces should compose these rather than inventing new control styles.

## Layout

The app shell uses a top navigation bar with a two-mode workbench below it. The processing screen prioritizes source controls, media evidence, and results. History appears as a drawer/workspace rather than a disconnected full-screen modal. Monitor mode shares the same panel and status vocabulary as processing mode.

## Motion

Transitions are 150-250ms and communicate state only: hover, focus, drawer entry, loading, and progress. `prefers-reduced-motion` disables animation and smooth scrolling.
