"""
build.py
This script transcribes MP3 files in the current directory using WhisperX and performs speaker diarization.
It saves the output in a specified folder with timestamps and speaker labels.
"""

# Import from Python standard library
from datetime import timedelta
import time
import os

# Import third-party libraries
import torch
import whisperx

# Define paths
root_folder = "."
output_folder = "transcripts"
os.makedirs(output_folder, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
print(f"Running on device: {device} with compute type: {compute_type}")

print("Loading WhisperX model. Please be patient. ...")
model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
print("WhisperX model loaded.")

# Helper function to format time nicely
def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(timedelta(seconds=round(seconds)))

# Iterate over MP3 files in the root
for filename in os.listdir(root_folder):
    if filename.endswith(".mp3"):
        print(f"\nStarting transcription for {filename}...")
        start_time = time.time()

        # Transcribe with WhisperX (no device parameter needed here)
        filepath = os.path.join(root_folder, filename)
        result = model.transcribe(filepath)

        # 1. Align the transcriptions
        alignment_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], alignment_model, metadata, filepath, device, return_char_alignments=False)
        print("Transcription and alignment completed.")

        # Save the aligned transcript
        aligned_txt = os.path.join(output_folder, f"{filename}_aligned.txt")
        with open(aligned_txt, "w") as f:
            for segment in result["segments"]:
                f.write(f"{segment['start']} - {segment['end']}: {segment['text']}\n")      

        # Load the Diarization Pipeline
        diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=False)
        print("Diarization pipeline loaded.")

        # Perform speaker diarization
        diarize_segments = diarize_model(filepath)
        diarize_segments = diarize_segments.cpu().numpy()

        # Add speaker labels to the aligned segments
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Prepare output path
        diarized_txt = os.path.join(output_folder, f"{filename}_diarized.txt")

        # Create a dictionary to map speakers
        speaker_mapping = {}
        speaker_count = 1

        # Save the diarized transcript
        with open(diarized_txt, "w") as f:
            for segment in diarize_segments:
                start, end, speaker = segment

                # Map speaker IDs to Speaker 1, Speaker 2, etc.
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"Speaker {speaker_count}"
                    speaker_count += 1
                
                speaker_label = speaker_mapping[speaker]

                # **Get only the text within the timestamp range**
                segment_text = model.extract_text(result, start, end)

                # Write with formatted timestamp and speaker label
                f.write(f"[{format_timestamp(start)} - {format_timestamp(end)}] {speaker_label}:\n")
                f.write(f"{segment_text}\n\n")

        end_time = time.time()
        print(f"Finished transcription for {filename} in {round(end_time - start_time, 2)} seconds.")
        print(f"Saved to {diarized_txt}")
