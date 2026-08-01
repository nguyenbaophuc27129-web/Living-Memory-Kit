"""
clean_audio.py
--------------
Lọc nhiễu nền (tiếng quạt, tiếng ù điện, tiếng phố xá) khỏi các file ghi âm
phỏng vấn thô, dùng thuật toán dựa trên FFT (qua thư viện noisereduce).

Dùng cho TỪNG file ghi âm trước khi cắt đoạn và đưa vào dataset.

Cách chạy:
    python clean_audio.py raw_audio/phong_van_bac_ba.wav audio/demo_001_clean.wav
"""

import sys
import noisereduce as nr
import soundfile as sf


def clean_audio(input_path: str, output_path: str):
    print(f"Đang đọc: {input_path}")
    audio, sr = sf.read(input_path)

    print("Đang phân tích phổ tần số và lọc nhiễu (FFT/STFT)...")
    # noisereduce tự động ước lượng nhiễu nền từ chính đoạn audio
    # (stationary=True phù hợp với nhiễu ổn định như tiếng quạt, ù điện)
    cleaned = nr.reduce_noise(y=audio, sr=sr, stationary=True)

    sf.write(output_path, cleaned, sr)
    print(f"Đã lưu bản đã lọc nhiễu tại: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python clean_audio.py <file_dau_vao.wav> <file_dau_ra.wav>")
        sys.exit(1)
    clean_audio(sys.argv[1], sys.argv[2])
