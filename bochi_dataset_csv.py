import os
import csv
import whisper

model = whisper.load_model("large")
def transcribe_audio_to_csv(directory):
    # Load the whisper model
    # Get a list of all .wav files in the directory
    wav_files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])
    # Open the CSV file for writing
    with open('metadata.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter='|')

        # Process each .wav file
        for wav_file in wav_files:
            # Transcribe the audio file
            result = model.transcribe(os.path.join(directory, wav_file), language="ja", fp16=False)
            transcribed_text = result["text"]

            # Write the filename and transcribed text to the CSV file
            print(f"Transcribed {wav_file}: {transcribed_text}")
            writer.writerow([wav_file, transcribed_text ,transcribed_text])

# Use the function
transcribe_audio_to_csv('./wavs')