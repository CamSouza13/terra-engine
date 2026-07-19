# Sidebar background asset

Drop a single file here to give the console sidebar a background:

- `sidebar.mp4` — a looping video (best; generate in Sora), or
- `sidebar.jpg` — a still image (used as the video poster / fallback).

The console requests `/assets/sidebar.mp4` (with `poster="/assets/sidebar.jpg"`).
If neither exists, the sidebar falls back to its built-in dark starfield gradient,
so nothing breaks. A dark, slow, low-detail clip works best — the nav text sits on
top with a scrim, and busy footage hurts legibility.

After adding a file, redeploy (`fly deploy`) so it ships in the image.
