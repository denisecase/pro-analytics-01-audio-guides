"""
build2.py
This script transcribes MP3 files in the current directory using WhisperX and performs speaker diarization.
It saves the output in a specified folder with timestamps and speaker labels, handling errors gracefully.
"""

# Import from Python standard library
from datetime import timedelta
import time
import os
import logging

# Import third-party libraries
import torch
import whisperx

# Logging configuration
logging.basicConfig(filename='transcription_errors.log',
                    level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
root_folder = "."
output_folder = "transcripts2"
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

# Helper function to write to the transcript
def write_to_transcript(f, start, end, speaker, text):
    f.write(f"[{format_timestamp(start)} - {format_timestamp(end)}] {speaker}:\n")
    f.write(f"{text}\n\n")

# Process each MP3 in batches
def process_in_batches(filepath):
    print(f"\nStarting batch processing for {filepath}...")
    start_time = time.time()
    
    diarized_txt = os.path.join(output_folder, f"{os.path.basename(filepath)}_diarized.txt")
    aligned_txt = os.path.join(output_folder, f"{os.path.basename(filepath)}_aligned.txt")
    
    try:
        # Transcribe with WhisperX
        result = model.transcribe(filepath)

        # Align the transcriptions
        alignment_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], alignment_model, metadata, filepath, device, return_char_alignments=False)
        print("Transcription and alignment completed.")

        # Save the aligned transcript
        with open(aligned_txt, "w") as f:
            for segment in result["segments"]:
                f.write(f"{segment['start']} - {segment['end']}: {segment['text']}\n")

        # Load the Diarization Pipeline
        try:
            diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=False)
            print("Diarization pipeline loaded.")

            # Perform speaker diarization
            diarize_segments = diarize_model(filepath)
            diarize_segments = diarize_segments.cpu().numpy()

            # Create a dictionary to map speakers
            speaker_mapping = {}
            speaker_count = 1

            with open(diarized_txt, "w") as f:
                for segment in diarize_segments:
                    start, end, speaker = segment
                    if speaker not in speaker_mapping:
                        speaker_mapping[speaker] = f"Speaker {speaker_count}"
                        speaker_count += 1

                    speaker_label = speaker_mapping[speaker]
                    segment_text = model.extract_text(result, start, end)
                    write_to_transcript(f, start, end, speaker_label, segment_text)
            
            print(f"Finished processing for {filepath} in {round(time.time() - start_time, 2)} seconds.")
            print(f"Data saved to {diarized_txt}")

        except Exception as e:
            print(f"❌ Diarization failed for {filepath}: {e}")
            logging.error(f"Diarization failed for {filepath}: {e}")

    except Exception as e:
        print(f"❌ Transcription failed for {filepath}: {e}")
        logging.error(f"Transcription failed for {filepath}: {e}")

# Loop through MP3 files in batches
for filename in os.listdir(root_folder):
    if filename.endswith(".mp3"):
        process_in_batches(os.path.join(root_folder, filename))
