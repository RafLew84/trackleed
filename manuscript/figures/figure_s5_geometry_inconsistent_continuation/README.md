# Figure S5 — Geometry-inconsistent continuation examples

Status: generated and validated.

Figure S5 shows one deterministic, source-linked example per method. For each
method in the frozen display order, the freeze step selects the first accepted
Gaussian fit outside the 18 px branch-consistency gate in frozen
case/frame/branch order. It never selects by error magnitude. Each committed
crop is hash-linked to both its measurement row and raw image.

**Caption.** **a–e,** LocoTrack, OSTrack, DAM4SAM, Lucas–Kanade and EasyLEED
examples, respectively. The black cross is the frozen expected branch
coordinate, the colored triangle is the native tracking proposal, and the
white circle is the accepted common Gaussian center. **f,** Selection rule and
the full method-specific geometry-inconsistent/accepted-fit counts. The
examples are operational evidence of possible false continuity. They are not
externally proven identity switches, do not constitute a ranking and are not
selected to maximize a method's branch error.

```bash
python freeze_source_data.py --validate-only
python build_figure.py --validate-only
python -m unittest test_build_figure.py
```
