#!/usr/bin/env python3
"""
MCP Server for Speech-to-Text using Whisper and FastMCP with Voice Activity Detection (VAD)
"""

import os
import sys
import tempfile
import time
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from fastmcp import FastMCP
import webrtcvad
import io
import contextlib

# --- Configuration ---
MODEL_TYPE = "medium"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
VAD_MODE = 3  # 0: very sensitive, 3: very aggressive
FRAME_DURATION = 30  # ms
SILENCE_TIMEOUT = 3.0  # seconds of silence to consider end of speech
MIN_AUDIO_AMPLITUDE_THRESHOLD = 0.05 # Minimum amplitude to consider audio as speech
DEBUG_MODE = False # Set to False to disable debug logging

# Load the Whisper model once at startup
WHISPER_MODEL = None
try:
    # Determine the device (GPU if available, otherwise CPU) 
    # mps doesn't work well with whisper, so we use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the Whisper model
    WHISPER_MODEL = whisper.load_model(MODEL_TYPE, device=device)
except Exception as e:
    # Print error and exit if model loading fails
    print(f"Error loading Whisper model: {e}", file=sys.stderr)
    sys.exit(1)

# Create the FastMCP server instance for the "speech" service
mcp = FastMCP("speech")

def record_audio_vad():
    """
    Records audio from the default microphone using VAD to detect speech start and end.
    The recording stops after a period of silence.
    Returns a numpy array of the recorded audio.
    """
    # Initialize the VAD with the configured mode
    vad = webrtcvad.Vad(VAD_MODE)
    # Calculate frame length in samples
    frame_length = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    # Calculate silence limit in frames
    silence_limit = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)
    audio_buffer = [] # Buffer to store recorded audio frames
    silence_counter = 0 # Counter for consecutive silence frames
    speech_started = False # Flag to indicate if speech has been detected
    start_time = time.time() # To track overall recording duration without speech

    def callback(indata, frames, time_info, status):
        """
        Callback function for sounddevice.InputStream.
        Processes incoming audio data for VAD and buffering.
        
        Only append `indata` to `audio_buffer` when `is_speech` is True.
        When `is_speech` is False, we only increment `silence_counter` if `speech_started` is True.
        This prevents buffering large segments of silence when speech is not present.
        """
        nonlocal silence_counter, speech_started
        if DEBUG_MODE:
            print("Callback triggered.")
        
        # Convert incoming float32 audio data to 16-bit PCM bytes, required by webrtcvad
        pcm_data = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        
        # Check if the current frame contains speech
        is_speech = vad.is_speech(pcm_data, SAMPLE_RATE)
        
        if DEBUG_MODE:
            print(f"Is speech: {is_speech}, Silence counter: {silence_counter}, Speech started: {speech_started}, Max Amplitude: {np.max(np.abs(indata)):.4f}")
        
        if is_speech:
            # If speech is detected, reset silence counter and add the speech frame to buffer
            if not speech_started and DEBUG_MODE:
                print("Speech started!")
            audio_buffer.append(indata.copy()) # Append actual speech frame
            silence_counter = 0 # Reset silence counter as speech is detected
            speech_started = True
        else: # Current frame is not speech
            if speech_started: # If speech was previously detected
                silence_counter += 1 # Increment silence counter
                # IMPORTANT: We DO NOT append `indata.copy()` here.
                # This ensures that only speech frames are buffered, minimizing silent audio passed to Whisper.
            # If speech_started is False and current frame is not speech, do nothing (wait for speech).

    # Start the audio input stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=frame_length, callback=callback, device=0):
            if DEBUG_MODE:
                print("Listening for speech...")
            
            # Keep the stream open until silence limit is reached after speech has started
            # or until speech is detected for the first time.
            while True:
                sd.sleep(FRAME_DURATION) # Sleep for the duration of one frame
                if speech_started and silence_counter >= silence_limit:
                    if DEBUG_MODE:
                        print("Silence limit reached, ending recording.")
                    break
                # If speech hasn't started, and we're not accumulating silence, just keep sleeping
                # The loop will break only when speech_started is true AND silence_counter hits the limit.
    except Exception as e:
        print(f"Error during audio recording: {e}", file=sys.stderr)
        return np.zeros((1, CHANNELS), dtype=np.float32) # Return empty array on error

    # Concatenate all recorded audio frames if any were captured
    if audio_buffer:
        if DEBUG_MODE:
            print(f"Recorded audio buffer length: {len(audio_buffer)}")
        return np.concatenate(audio_buffer, axis=0)
    else:
        if DEBUG_MODE:
            print("No audio recorded.")
        return None # Changed from np.zeros to None

#Noticed that Cursor sometimes doens't load the tool if there are no parameters.
@mcp.tool()
def speech_to_text(_: str = "") -> str:
    """
    Listens to the user's speech, transcribes it using Whisper, and returns the text.
    Includes extensive debugging output if DEBUG_MODE is True.
    `_` is a placeholder to satisfy tool schema.
    """
    output_filename = None
    transcribed_text = ""
    debug_log = ""

    # Redirect stdout to capture debug logs if DEBUG_MODE is True
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    if DEBUG_MODE:
        sys.stdout = redirected_output

    try:
        # Record audio using VAD
        audio_data = record_audio_vad()

        # Check if any significant audio was recorded
        if audio_data is not None and audio_data.size > 0 and np.max(np.abs(audio_data)) > MIN_AUDIO_AMPLITUDE_THRESHOLD:
            # Create a temporary WAV file to save the recorded audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                output_filename = tmpfile.name
                # Write the audio data to the temporary WAV file
                wav.write(output_filename, SAMPLE_RATE, audio_data)
            
            # Get the loaded Whisper model
            model = WHISPER_MODEL
            # Transcribe the audio file using Whisper
            result = model.transcribe(output_filename)
            transcribed_text = result["text"]
        else:
            transcribed_text = "No speech detected."

        if DEBUG_MODE:
            # Restore original stdout and include captured debug log in the response
            sys.stdout = old_stdout
            debug_log = redirected_output.getvalue()
            return f"Transcription: {transcribed_text}\nDebug Log:\n{debug_log}"
        else:
            # Return only the transcription if debug mode is off
            return transcribed_text
    except Exception as e:
        # Handle exceptions during transcription process
        if DEBUG_MODE:
            # Restore original stdout on error and include debug log
            sys.stdout = old_stdout
            debug_log = redirected_output.getvalue()
            return f"Error: {e}\nDebug Log:\n{debug_log}"
        else:
            # Return only the error message if debug mode is off
            return f"Error: {e}"
    finally:
        # Ensure the temporary audio file is removed
        if output_filename and os.path.exists(output_filename):
            os.remove(output_filename)

if __name__ == "__main__":
    # Check if the Whisper model was loaded successfully before running the MCP server
    if WHISPER_MODEL is None:
        print("Whisper model not loaded. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Run the FastMCP server
    mcp.run()
