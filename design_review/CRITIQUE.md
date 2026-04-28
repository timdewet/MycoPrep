# Visual Critique — ImagePipelineGUI redesign (post-implementation)

Captured 2026-04-25. Screenshots in [screens/](screens). 5 stages × 2 themes.

Issues are tagged **[BUG]** (broken behaviour), **[VIS]** (visual / styling), **[UX]** (information architecture), and ranked P1 (fix soon) / P2 (nice-to-have).

---

## P1 — Bugs that need fixing

### 1. [BUG] Theme pill label doesn't update on programmatic theme change
**Where:** all `*_dark.png` — pill in the header still reads "☀ Light".
**Why:** `_sync_theme_pill()` in [main_window.py](src/image_pipeline_gui/main_window.py) is only called on user click and on startup; programmatic `set_theme_override()` (and the system-theme follower) bypass it.
**Fix:** add a `theme.add_theme_listener(...)` registration in `_build_chrome()` that re-runs `_sync_theme_pill()` whenever the palette changes.

### 2. [BUG] Breadcrumb stuck at "Step 1 of 1"
**Where:** every screenshot — header reads "Step 1 of 1 · Input" / "…· Run" etc.
**Why:** `_on_nav_changed` in [main_window.py:191-203](src/image_pipeline_gui/main_window.py:191) counts visible entries via `self._stack.widget(i).isHidden()`. A `QStackedWidget` reports every non-current page as hidden, so the count always collapses to 1.
**Fix:** track visible keys separately (the sidebar already maintains `self._visible_keys`); reuse that set rather than asking the stack widgets.

### 3. [BUG] Sidebar icons are missing for Input, Plate, Run
**Where:** [input_light.png](design_review/screens/input_light.png), all dark frames — Input/Run rows have empty icon padding; Plate row's icon also doesn't render.
**Why:** glyph names in [icons.py:24-41](src/image_pipeline_gui/ui/icons.py:24) include some that aren't valid in the qtawesome MDI set: `mdi.tray-arrow-down`, `mdi.shape-outline`, `mdi.image-search-outline`, `mdi.arrow-down-thin-circle-outline`. (`mdi.target` and `mdi.brain` rendered fine, which is why Focus and Segment rows are OK.)
**Fix:** swap to confirmed-good MDI names — e.g. `mdi.upload`, `mdi.shape`, `mdi.image-search`, `mdi.arrow-down-circle-outline`. Add a sanity log when `qta.icon` raises so future bad names surface.

### 4. [BUG] Plate-layout page renders even when sidebar item is hidden
**Where:** [plate_dark.png](design_review/screens/plate_dark.png) — sidebar shows no Plate row (Bulk mode), but the stack still draws the plate when index 1 is selected.
**Why:** `_on_input_mode_changed` hides only the sidebar item; the stack page is left visible.
**Fix:** when bulk-hiding plate, also `setEnabled(False)` on the page and skip it when restoring `nav_index` (or reindex on visibility changes).

### 5. [BUG] "Channels" header rendered as orphan label outside any card
**Where:** [input_light.png](design_review/screens/input_light.png), [input_dark.png](design_review/screens/input_dark.png) — bold "Channels" label floats with no surrounding card; visually inconsistent with "Input mode" / "Output" / "Bulk CZI batch" which all sit inside cards.
**Why:** [input_panel.py](src/image_pipeline_gui/panels/input_panel.py) emits a `QLabel#sectionHeader` then a `QGroupBox` underneath.
**Fix:** drop the standalone label and use the group box title for "Channels" like the other sections.

### 6. [BUG] Form-row labels render with dark rectangles in dark mode
**Where:** [focus_dark.png](design_review/screens/focus_dark.png), [input_dark.png](design_review/screens/input_dark.png) — "Mode:" / "Metric:" / "Phase channel:" / "Output dir:" labels and the checkbox text rows show subtle dark fills behind the label cells.
**Why:** `QFormLayout` adds a wrapping container per row; my `* { color: ... }` rule plus the global `QMainWindow, QWidget { background: bg }` cascade through, but the form cell takes its background from the `QGroupBox` parent which is `surface` — they shouldn't differ. The artifact is most likely the row label-cell being a `QLabel` whose alignment area picks up `QToolTip`-like styling. Worth pinning down with `QLabel { background: transparent; }` in the QSS.
**Fix:** add `QLabel { background: transparent; }` to [theme.py](src/image_pipeline_gui/ui/theme.py) and confirm it disappears.

### 7. [BUG] Run-stage status dot stuck amber after my smoke-test driver
Cosmetic-only in screenshots, but represents a real risk: the sidebar's status dot is amber on Run even after the run wraps, because we set `RUNNING` in `_on_stage_run_started` and never clear it back to IDLE on stage finish (only on full run finish).
**Fix:** in the running-state cleanup at the end of `_on_run_requested`, reset Run's status to DONE on success / ERROR on fail (already done) — but also handle the case where the run is cancelled. Verify the post-run state matches reality.

---

## P2 — Visual / UX

### 8. [VIS] Cards stretch to fill the window when content is short
**Where:** [focus_light.png](design_review/screens/focus_light.png), [focus_dark.png](design_review/screens/focus_dark.png), [input_*.png](design_review/screens/input_light.png) — the Focus card is mostly empty grey; the Output card has 1 row of content in a card 200px tall.
**Why:** the stack-page wrapper in [main_window.py:_build_chrome()](src/image_pipeline_gui/main_window.py) doesn't add a stretch after the panel content, so the panel fills available height.
**Fix:** wrap the panel in a `QVBoxLayout` that adds `addStretch(1)` after the panel, or have each panel's root layout end in a stretch (most already do — but `FocusPanel`'s top-level QVBox is the panel itself and the stretch is internal).

### 9. [VIS] Output card is a giant empty area with one right-aligned row
**Where:** [input_light.png](design_review/screens/input_light.png).
**Why:** `QFormLayout` with one row puts the field flush to the right of the row, leaving the entire left column empty.
**Fix:** drop the `QFormLayout` for this single-row case and use a horizontal layout with `Output dir:` label, edit, browse button — left-aligned.

### 10. [UX] "Reuse existing outputs" sits inside the Stages-to-run card
**Where:** [run_light.png](design_review/screens/run_light.png), [run_dark.png](design_review/screens/run_dark.png).
**Why:** I tucked it on the right of the same row as the stage checkboxes during the refit. Conceptually it's a runtime modifier of the whole run, not a stage selection.
**Fix:** move it into the sticky action bar, between the Stop button and the Open output folder button. The action bar already has space.

### 11. [UX] Header lacks an anchor / brand
**Where:** every screenshot — header has only the breadcrumb (small, far left) and a theme pill (far right), with the entire centre and a third of the left empty. The "Image Pipeline" brand sits in the sidebar instead, where it competes with the active-step indicator.
**Fix:** move the brand to the header (keep the sidebar's first row clean — selected stage is the visual anchor there) **or** add a simple breadcrumb arrow design ("Input › Plate layout › Focus") that fills more of the header.

### 12. [VIS] Empty plate wells too dark in dark mode
**Where:** [plate_dark.png](design_review/screens/plate_dark.png) — the empty wells fall back to `well_empty_fill = #1c222b`, only marginally lighter than the body bg `#0e1116`. They almost disappear.
**Fix:** lift `well_empty_fill` in dark to ~`#242b35` and use `well_empty_ring = #4a5260` for a stronger silhouette.

### 13. [VIS] Brand label competes with selection bar in the sidebar
**Where:** all screenshots — "Image Pipeline" brand top-left of sidebar uses `#brand` (h2 size, bold) which is heavier than any nav item label, even the selected one.
**Fix:** if it stays in the sidebar, drop to label size with `text_subtle` colour, or replace it with a small product icon.

### 14. [VIS] Stop button danger styling invisible until pressed
The `#danger` red only shows up while the button is hot. When disabled (the common state) it's the standard greyed-out look — indistinguishable from any other secondary button. The clear "this is destructive" affordance is lost in the calm state.
**Fix:** keep a faint red tint on the icon when disabled, or use a red outline that softens but stays present.

### 15. [VIS] Missing combobox drop-down indicator
**Where:** all forms — the combo boxes (Mode/Metric/Cellpose model, etc.) read like read-only inputs. The `QComboBox::drop-down { border: none; width: 20px; }` rule removes the native arrow without supplying a glyph.
**Fix:** add a chevron icon via `QComboBox::down-arrow { image: url(<chevron.svg>); }` (use a tokenised SVG written to temp like the check icon).

### 16. [VIS] QSpinBox arrows look small / native
**Where:** [segclass_light.png](design_review/screens/segclass_light.png) — the up/down arrows next to numeric fields default to OS native chrome and look out of place against the otherwise tokenised app.
**Fix:** style `QSpinBox::up-button` / `::down-button` in [theme.py](src/image_pipeline_gui/ui/theme.py) with our tokens, or hide them and let the user type/use the wheel.

### 17. [VIS] Footer GPU indicator hard to read
"GPU unavailable (CPU)" sits at the bottom of the sidebar in `caption` size + `text_subtle` colour. On the lightest grey background it's barely there; on dark it fares better.
**Fix:** raise to `text_muted`, possibly add a coloured dot for status (red = unavailable, green = available).

### 18. [VIS] Spacing artifact in icon-prefixed buttons
**Where:** several buttons use a leading "  " (two spaces) hack to leave room for the icon (e.g. "  Run pipeline", "  Browse"). On metric-aware fonts this is fragile and visible as inconsistent gap.
**Fix:** drop the leading whitespace; rely on Qt's icon spacing (which already inserts a gap between icon and text). If the gap is too tight, set `iconSize` and bump button padding.

### 19. [VIS] Tab/keyboard focus rings not styled
None of the screenshots show focus state, but the QSS doesn't define `:focus` for buttons or the segmented control. Keyboard users will see Qt's default dotted ring, which clashes with the modern look.
**Fix:** add `QPushButton:focus { outline: 2px solid <focus_ring>; outline-offset: 1px; }` (or use border tricks, since QSS doesn't support outline-offset).

### 20. [VIS] No visible hover state on segmented control
The segmented control buttons have a checked state but no hover (you only see the active selection, never a "this is hoverable" cue).
**Fix:** the QSS already has `QPushButton#segLeft:hover:!checked` etc. but the styling is very subtle — bump to a clearer `surface_alt` fill plus an `text_muted → text` colour shift.

### 21. [UX] Plate column 12 obscured by the right-side editor in default sizing
**Where:** [plate_dark.png](design_review/screens/plate_dark.png) — only columns 1–11 visible; column 12 is cut off where the editor pane begins. There's a horizontal scrollbar but no visual hint that more wells exist.
**Fix:** either give the plate a horizontal scroll affordance, shrink wells when plate width < required, or make the right editor a dialog/inspector that doesn't permanently steal width.

### 22. [VIS] Top-bar "Step …" sits in a slightly differently-coloured rectangle
**Where:** every screenshot top-left — the breadcrumb has a subtly darker rectangular background that doesn't extend the full header. Looks like the focus rectangle on a label that gained focus.
**Fix:** verify the breadcrumb label has no implicit selection / focus, set `QLabel { background: transparent; }` globally.

---

## What's working well

Worth calling out so we don't regress:

- **Run page stepper** ([run_light.png](design_review/screens/run_light.png)) — numbered badges + per-stage progress strips read instantly; the active step's blue badge with an underlined fill is exactly the visual cue I wanted.
- **Log color-coding** is legible in both themes; warning amber and error red are clear without being garish.
- **Dark mode overall** — the GitHub-inspired palette holds up across all five stages without any token feeling wrong.
- **Plate map** — the well rendering is gorgeous; condition fills + reporter rings + sub-text all read at a glance.
- **Sticky run/stop action bar** at the bottom of the run page makes the primary CTAs unmissable.
- **Toolbar cards** in the Plate layout and Segment&Classify pages group their actions clearly.

---

## Suggested fix order

If implementing in one pass:

1. Fix the four bugs (#1, #2, #3, #6) — small, isolated, high-impact.
2. Tighten card heights (#8, #9) — one shared layout pattern, applied everywhere.
3. Add the missing form polish (#15, #16, #19) — all live in [theme.py](src/image_pipeline_gui/ui/theme.py).
4. Information-architecture nudges (#10, #11, #21).
5. Token tweaks (#12, #17).

Estimate: a focused half-day for everything in P1 + half of P2.
