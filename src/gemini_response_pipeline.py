"""
Authors: 
    Steven Liem (steven.liem@sydney.edu.au)

This script is used to generate a corpus of speech data from the Heet dataset using the Gemini 3.1 Flash Live Preview model.

Usage:
    uv run python gemini_response_pipeline.py
    uv run python gemini_response_pipeline.py --input <input_path> --output-dir <output_dir> --dry-run --limit <limit>
    
    Note: You can re-run the script to after the first run to continue from the next file. The script will automatically skip files that 
    already exist in the folderand continue from the next file. To restart, simply delete the output directory or create a new directory, and then run the script again.

Arguments:
    --input: Path to the input CSV file containing the questions and emotions.
    --output-dir: Path to the output directory where the generated speech data will be saved.
    --dry-run: If set, the script will run in dry mode and only generate the first 10 speech files.
    --limit: Maximum number of speech files to generate.

References:
    https://github.com/google-gemini/gemini-live-api-examples
    https://ai.google.dev/gemini-api/docs/live-guide#audio-transcription

"""

import argparse
import asyncio
import os
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from scipy import signal

MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_INPUT = "/home/steve/tmp/Monash Research Code/monash-research-tone-bias/heet_dataset_clean.csv"
DEFAULT_OUTPUT_DIR = "/home/steve/tmp/Monash Research Code/our_speech_corpus_test"

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")


def save_pcm_to_16khz_wav(pcm_bytes: bytes, filename: str | Path) -> None:
    """Re-sample the Gemini model's audio output to a 16kHz WAV file. This is required because the Gemini model outputs 24kHz WAV files

    Args:
        pcm_bytes: The Gemini model's audio output to save.
        filename: The filename to save the WAV file to.

    Returns:
        None
    """
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    num_samples = int(len(audio_data) * 16000 / 24000)
    resampled_data = signal.resample(audio_data, num_samples).astype(np.int16)
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(resampled_data.tobytes())


async def generate_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    limit: int | None,
    checkpoint_every: int = 25,
) -> None:
    """Generate Gemini speech responses from the cleaned HEET dataset.

    Args:
        input_path: Path to the input CSV file containing the questions and emotions.
        output_dir: Path to the output directory where the generated speech data will be saved.
        limit: Maximum number of speech files to generate.
        checkpoint_every: Number of speech files to generate before writing to the CSV file.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    work = df.head(limit) if limit is not None else df

    client = genai.Client(api_key=API_KEY)

    # Live API supports one primary response modality. Keep AUDIO and enable
    # output_audio_transcription to stream the spoken reply as text.
    # See: https://ai.google.dev/gemini-api/docs/live-guide#audio-transcription
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": "You are Gemini, an AI personal assistant. Answer the user's question in **exactly one short sentence** in under 7 seconds."}]},
    }

    success = 0
    failed = 0
    skipped = 0
    for i, row in work.iterrows():
        question_number = i + 1
        emotion = row["emotion_label"]
        bin_label = row["valence_label"]
        prompt = f"The user is currently speaking in a **{emotion}** tone: {row['Question']}"

        # Format the filename as: <question_number>_<emotion>_<bin_label>.wav
        filename = f"{question_number}_{emotion}_{bin_label}.wav"
        filepath = output_dir / filename

        if filepath.exists() and df.at[i, "response_text"]:
            df.at[i, "audio_path"] = str(filepath)
            skipped += 1
            continue

        try:
            # Main entry point for the Gemini model. Check the references for more details.
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                await session.send(input=prompt, end_of_turn=True)

                audio_chunks = bytearray()
                transcript_chunks: list[str] = []
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content is None:
                        continue

                    model_turn = server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data:
                                audio_chunks.extend(part.inline_data.data)

                    # Streamed transcript of the model's audio output
                    output_transcription = server_content.output_transcription
                    if output_transcription and output_transcription.text:
                        transcript_chunks.append(output_transcription.text)

                    if server_content.turn_complete:
                        break

            response_text = "".join(transcript_chunks).strip()

            if audio_chunks:
                save_pcm_to_16khz_wav(audio_chunks, filepath)
                df.at[i, "audio_path"] = str(filepath)
                df.at[i, "response_text"] = response_text
                success += 1
                print(f"Saved: {filename}")
            else:
                failed += 1
                print(f"No audio for {filename}")

        except Exception as e:
            failed += 1
            print(f"Failed {filename}: {e}")

        # Write to CSV file every 25 speech files. This checkpoint mechanism is designed in case of failure during long running tasks.
        if (success + failed) % checkpoint_every == 0:
            df.to_csv(input_path, index=False)
            print(f"Checkpoint: wrote {input_path}")

        # Sleep for 1 second to avoid Gemini API rate limits. This can be eliminated if you have a paid Gemini API account.
        await asyncio.sleep(1)

    df.to_csv(input_path, index=False)
    print(f"Done: {success} saved, {failed} failed, {skipped} skipped to {output_dir}")
    print(f"Updated manifest: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 10 if args.dry_run else args.limit # Dry run default to generating 10 speech files.
    asyncio.run(generate_corpus(args.input, args.output_dir, limit))


if __name__ == "__main__":
    main()
