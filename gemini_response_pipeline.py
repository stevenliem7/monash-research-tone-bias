"""
This script is used to generate a corpus of speech data from the Heet dataset using the Gemini 3.1 Flash Live Preview model.

Usage:
    python gemini_tts_pipeline.py --input <input_path> --output-dir <output_dir> --dry-run --limit <limit>
    Dry run: 

Arguments:
    --input: Path to the input CSV file containing the questions and emotions.
    --output-dir: Path to the output directory where the generated speech data will be saved.
    --dry-run: If set, the script will run in dry mode and only generate the first 10 speech files.
    --limit: Maximum number of speech files to generate.
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from google import genai

MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_INPUT = "/home/steve/tmp/Monash Research Code/monash-research-tone-bias/heet_dataset_clean.csv"
DEFAULT_OUTPUT_DIR = "/home/steve/tmp/Monash Research Code/our_speech_corpus"

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")

def save_pcm_to_16khz_wav(pcm_bytes, filename):
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    num_samples = int(len(audio_data) * 16000 / 24000)
    resampled_data = signal.resample(audio_data, num_samples).astype(np.int16)
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(resampled_data.tobytes())


async def generate_corpus(input_path, output_dir, limit, checkpoint_every=25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "audio_path" not in df.columns:
        df["audio_path"] = ""

    work = df.head(limit) if limit is not None else df

    client = genai.Client(api_key=gemini_api_key)

    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": "You are Gemini, an AI personal assistant. Answer the user's question in **exactly one short sentence** in under 7 seconds."}]},
    }

    success = 0
    failed = 0
    for i, row in work.iterrows():
        question_number = i + 1
        emotion = row["emotion_label"]
        bin_label = row["valence_label"]
        prompt = f"The user is currently speaking in a **{emotion}** tone: {row['Question']}"
        filename = f"{question_number}_{emotion}_{bin_label}.wav"
        filepath = output_dir / filename

        try:
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                await session.send(input=prompt, end_of_turn=True)

                audio_chunks = bytearray()
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content is not None:
                        model_turn = server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.inline_data:
                                    audio_chunks.extend(part.inline_data.data)
                        if server_content.turn_complete:
                            break

            if audio_chunks:
                save_pcm_to_16khz_wav(audio_chunks, str(filepath))
                df.at[i, "audio_path"] = str(filepath)
                success += 1
                print(f"Saved: {filename}")
            else:
                failed += 1
                print(f"No audio for {filename}")

        except Exception as e:
            failed += 1
            print(f"Failed {filename}: {e}")

        if (success + failed) % checkpoint_every == 0:
            df.to_csv(input_path, index=False)
            print(f"Checkpoint: wrote {input_path}")

        await asyncio.sleep(1)

    df.to_csv(input_path, index=False)
    print(f"Done: {success} saved, {failed} failed to {output_dir}")
    print(f"Updated manifest: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 10 if args.dry_run else args.limit
    asyncio.run(generate_corpus(args.input, args.output_dir, limit))


if __name__ == "__main__":
    main()
