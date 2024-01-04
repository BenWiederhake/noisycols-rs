# noisycols-rs

This makes pleasing, moving backgrounds "images", with bright and friendly colors, and a "fuzzy milkglass" pixel distortion overlay.

I already did something similar in 2019, in Python: https://github.com/BenWiederhake/noisycols#noisycols

My motivation this time was:
- Do it in Rust, because … get better at Rust.
- Create a video, because … pretty moving colors!

This eats up incredible amounts of computational power, so this must be multi-threaded in Rust.

## Table of Contents

- [Background](#background)
- [Usage](#usage)
- [Tweaking](#tweaking)
- [TODOs](#todos)
- [Contribute](#contribute)

## Background

![An example image, which also is a nice background image. There is nothing else in this section, because the entire section is a pun.](making-of/image_1704322316_frame00000.png)

## Usage

1. Tweak constants in `src/main.rs`
2. Generate images with `cargo run --release`
3. Compile the image sequence into a sharable video with: `ffmpeg -stream_loop 2 -y -f image2 -framerate 10 -i build/image_1234567890_frame%05d.png -vcodec libx264 -crf 22 -profile:v baseline -vf format=yuv420p -movflags +faststart /tmp/video.mp4`

The cargo step can be sped up even further by writing:
```
[build]
rustflags = ["-C", "target-cpu=native"]
```
into `.cargo/config.toml`

The reason for the long ffmpeg incantation is:
- `-stream_loop 2`: Not all video players support seamless looping, and this is a stop-gap measure to make it seamless at least half of the time.
- `-framerate` can of course be increased arbitrarily, but if you want a smoother (and not just faster) experience, then you'll also need to increase the constant `NUM_FRAMES` in `src/main.rs`.
- `1234567890` has to be replaced by the actual timestamp used during `cargo run`.
- `libx264`, `baseline`, `yuv240p`, `faststart`, and `.mp4` are necessary in order to support in-client video playback in Telegram, which is where I share these videos with friends.
- `crf 22` means quite good quality. Note that due to the nature of the video, the blobs move but the milkglass distortion does not move. This is probably an unexpected video signal that should be destroyed by compression, although in my experience it is actually handled quite well.

The result could look like this:

[Video of pleasing, colorful blobs](making-of/video_1704325712.mp4)

## Tweaking

Feel free to tweak the constants and experiment yourself! In the directory `making-of/` I put some (ugly) results that I got while choosing the right parameters.

## TODOs

Nothing, actually. I'm happy and "[done](https://thomasdeneuville.com/cult-of-done-manifesto/)" with this project.

## Contribute

Feel free to dive in! [Open an issue](https://github.com/BenWiederhake/noisycols-rs/issues/new) or submit PRs.
