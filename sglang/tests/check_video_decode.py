"""Check H.264 decoding through the installed SGLang video reader."""

from pathlib import Path

from sglang.srt.utils.video_decoder import VideoDecoderWrapper


def main():
    clip = Path(__file__).with_name("fixtures") / "frame-order.mp4"
    with VideoDecoderWrapper(clip.read_bytes()) as decoder:
        frames = decoder.get_frames_at([0, 12, 24])
    channels = frames.mean(axis=(1, 2)).argmax(axis=1).tolist()
    if channels != [2, 0, 1]:
        raise ValueError(f"Expected blue, red, green frames, got channels {channels}")
    print("Decoded H.264 frames in blue, red, green order")


if __name__ == "__main__":
    main()
