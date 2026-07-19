# Sidebar background — Sora prompts & install

The console sidebar is dark, tall, and narrow, with nav text and a scrim on top.
For legibility the ideal clip is **dark, slow, low-contrast, and loopable**, with
motion confined to the edges. Generate one in Sora, drop it in, redeploy.

## Specs

- **Aspect / size:** vertical, 1080×1920 (9:16). The sidebar is a tall column.
- **Length:** 10–20s, seamless loop.
- **Tone:** keep it dark. Bright or busy footage fights the white nav text.
- **File:** save as `sidebar.mp4` in `terra/web/assets/` (a `sidebar.jpg` still works too and is used as the poster/fallback).
- **Deploy:** `fly deploy` so it ships in the image.

## Space prompts

1. **Deep-space drift** — "A slow, silent drift through deep space. Sparse faint
   stars on a near-black field, a very subtle dark-blue nebula haze in the lower
   third, gentle parallax as if floating. Minimal, cinematic, high dynamic range
   but overall very dark. No text, no lens flares, seamless loop, vertical 9:16."

2. **Earth limb from orbit** — "The curve of Earth's dark night side seen from low
   orbit, a thin glowing blue atmospheric line along the edge, faint city lights,
   slow orbital motion. Mostly black frame with the horizon in the lower portion.
   Calm, premium, documentary. Vertical 9:16, seamless loop, no text."

3. **Quiet starfield** — "A static camera on a dense but dim starfield, almost
   imperceptible twinkle and the faintest slow rotation of the Milky Way band on
   one side. Deep charcoal-black, no bright objects. Meditative. Vertical 9:16,
   seamless loop."

## Nature prompts

4. **Dark water surface** — "Overhead view of very dark, calm water at night,
   subtle ripples catching a trace of moonlight, slow gentle motion. Deep teal-
   black, minimal, abstract. No horizon, no objects. Vertical 9:16, seamless loop,
   no text."

5. **Bioluminescent depths** — "Abstract underwater scene in near-black deep water
   with a few slow drifting motes of soft blue-green bioluminescence rising gently.
   Very dark, quiet, elegant — evokes a living system without being busy. Vertical
   9:16, seamless loop."

6. **Misted evergreen ridge** — "A dark forested ridge at dawn under heavy mist,
   muted deep greens and slate, slow-moving fog drifting across silhouetted
   evergreens. Low contrast, moody, restrained. Vertical 9:16, seamless loop, no
   text or people."

## Tips

- Add "very dark, low contrast, minimal, no text, no watermark, seamless loop" to
  any prompt — Sora respects those and it keeps the nav readable.
- If a clip is too bright once installed, either regenerate darker or reduce the
  media opacity in `index.html` (`.side-media { opacity: … }`, currently `0.5`).
- Match the theme to your lead domain: water/bioluminescence for aquaculture,
  evergreen/mist for soil, Earth-from-orbit for the BLSS / space story.
