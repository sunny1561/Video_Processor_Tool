# # import streamlit as st
# # from transformers import pipeline
# # import warnings
# # import librosa
# # from gtts import gTTS
# # from moviepy.editor import VideoFileClip
# # import ffmpeg
# # import os
# # from dotenv import load_dotenv
# # from openai import AzureOpenAI  # Make sure you have this installed

# # # Load environment variables from .env file
# # load_dotenv()

# # # Azure OpenAI client setup
# # client = AzureOpenAI(
# #     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
# #     api_key=os.getenv("AZURE_API_KEY"),
# #     api_version=os.getenv("AZURE_API_VERSION")
# # )

# # # Suppress specific future warnings
# # warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

# # # Load the Whisper model
# # whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# # def transcribe_audio(audio_file_path):
# #     """
# #     Function to transcribe an audio file using the Whisper model.
# #     """
# #     audio, sr = librosa.load(audio_file_path, sr=16000)
# #     transcription = whisper_model(audio)["text"]
# #     return transcription

# # def extract_audio(video_file):
# #     """
# #     Extracts the audio from a video file and saves it as an MP3 file.
# #     """
# #     audio_output = "extracted_audio.mp3"
# #     mp4_file = VideoFileClip(video_file)
# #     audio_only = mp4_file.audio
# #     audio_only.write_audiofile(audio_output)
# #     mp4_file.close()
# #     audio_only.close()
# #     return audio_output

# # def correct_transcription_azure(text):
# #     messages = [
# #         {
# #             "role": "user",
# #             "content": (
# #                 f"As a language correction assistant, your task is to improve the following transcription.\n"
# #                 f"1. Correct any grammatical mistakes.\n"
# #                 f"2. Remove filler words such as 'umm', 'hmm', and similar verbal pauses.\n"
# #                 f"3. Maintain the original meaning and intent of the transcription.\n"
# #                 f"4. Provide the corrected version without adding any new content or altering the original context.\n\n"
# #                 f"Original transcription:\n{text}"
# #             )
# #         }
# #     ]

# #     response = client.chat.completions.create(
# #         model="gpt-4o",
# #         messages=messages,
# #         max_tokens=1000  ## we can adjust it considering context length and output length
# #     )
# #     return response.choices[0].message.content.strip()

# # def text_to_speech(text):
# #     """
# #     Converts the given text to speech and saves it as an audio file.
# #     """
# #     tts = gTTS(text=text, lang='en')
# #     output_audio_file = "corrected.mp3"
# #     tts.save(output_audio_file)
# #     return output_audio_file

# # def replace_audio(video_path, new_audio_path):
# #     """
# #     Replaces the audio in the video with a new audio file.
# #     """
# #     output_video_path = "final_video.mp4"
# #     input_video = ffmpeg.input(video_path)
# #     input_audio = ffmpeg.input(new_audio_path)
# #     ffmpeg.output(input_video, input_audio, output_video_path, vcodec="copy", acodec="aac").run(overwrite_output=True)
# #     return output_video_path

# # # Streamlit app layout
# # st.title("Video Audio Processor")
# # st.write("Upload an MP4 video to process.")

# # uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

# # if uploaded_file is not None:
# #     # Save uploaded file
# #     video_path = "uploaded_video.mp4"
# #     with open(video_path, "wb") as f:
# #         f.write(uploaded_file.read())

# #     st.write("Processing the video...")

# #     # Extract audio
# #     audio_path = extract_audio(video_path)

# #     # Transcribe audio
# #     transcription = transcribe_audio(audio_path)

# #     # Correct transcription using Azure OpenAI
# #     corrected_trans = correct_transcription_azure(transcription)

# #     # Convert corrected text to speech
# #     tts_audio_path = text_to_speech(corrected_trans)

# #     # Replace audio in video
# #     final_video_path = replace_audio(video_path, tts_audio_path)

# #     # Download link for the final video
# #     with open(final_video_path, "rb") as f:
# #         st.download_button("Download Final Video", f, file_name="final_video.mp4", mime="video/mp4")


# import streamlit as st
# from transformers import pipeline
# import warnings
# import librosa
# from gtts import gTTS
# from moviepy.editor import VideoFileClip
# import ffmpeg
# import os
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# import time

# # Load environment variables from .env file
# load_dotenv()

# # Azure OpenAI client setup
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     api_key=os.getenv("AZURE_API_KEY"),
#     api_version=os.getenv("AZURE_API_VERSION")
# )

# # Suppress specific future warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

# # Load the Whisper model
# whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# def transcribe_audio(audio_file_path):
#     audio, sr = librosa.load(audio_file_path, sr=16000)
#     transcription = whisper_model(audio)["text"]
#     return transcription

# def extract_audio(video_file):
#     audio_output = "extracted_audio.mp3"
#     mp4_file = VideoFileClip(video_file)
#     audio_only = mp4_file.audio
#     audio_only.write_audiofile(audio_output)
#     mp4_file.close()
#     audio_only.close()
#     return audio_output

# def correct_transcription_azure(text):
    
#     messages = [
#         {
#             "role": "user",
#             "content": (
#                 f"As a language correction assistant, your task is to improve the following transcription.\n"
#                 f"1. Correct any grammatical mistakes.\n"
#                 f"2. Remove filler words such as 'umm', 'hmm', and similar verbal pauses.\n"
#                 f"3. Maintain the original meaning and intent of the transcription.\n"
#                 f"4. Provide the corrected version without adding any new content or altering the original context.\n\n"
#                 f"Original transcription:\n{text}"
#             )
#         }
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=1000
#     )
#     return response.choices[0].message.content.strip()

# def text_to_speech(text):
#     tts = gTTS(text=text, lang='en')
#     output_audio_file = "corrected.mp3"
#     tts.save(output_audio_file)
#     return output_audio_file

# def replace_audio(video_path, new_audio_path):
#     output_video_path = "final_video.mp4"
#     input_video = ffmpeg.input(video_path)
#     input_audio = ffmpeg.input(new_audio_path)
#     ffmpeg.output(input_video, input_audio, output_video_path, vcodec="copy", acodec="aac").run(overwrite_output=True)
#     return output_video_path

# # Streamlit app layout
# st.title("Welcome to Video Audio Processor")
# st.write("Upload an MP4 video to process.")

# def run_steps(video_path):
#     # Create placeholders for steps and progress bar
#     step_placeholder = st.empty()
#     progress_bar = st.progress(0)

#     # Step 1: Extract audio
#     step_placeholder.write("Step 1: Extracting audio from video...")
#     time.sleep(1)  # Simulate delay
#     audio_path = extract_audio(video_path)
#     progress_bar.progress(20)

#     # Step 2: Transcribe audio
#     step_placeholder.write("Step 2: Transcribing audio...")
#     time.sleep(1)  # Simulate delay
#     transcription = transcribe_audio(audio_path)
#     progress_bar.progress(40)

#     # Step 3: Correct transcription
#     step_placeholder.write("Step 3: Correcting transcription...")
#     time.sleep(1)  # Simulate delay
#     corrected_trans = correct_transcription_azure(transcription)
#     progress_bar.progress(60)

#     # Step 4: Convert text to speech
#     step_placeholder.write("Step 4: Converting corrected transcription to speech...")
#     time.sleep(1)  # Simulate delay
#     tts_audio_path = text_to_speech(corrected_trans)
#     progress_bar.progress(80)

#     # Step 5: Replace audio in video
#     step_placeholder.write("Step 5: Replacing audio in video...")
#     time.sleep(1)  # Simulate delay
#     final_video_path = replace_audio(video_path, tts_audio_path)
#     progress_bar.progress(100)

#     return final_video_path

# # File uploader
# uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

# if uploaded_file is not None:
#     # Save uploaded file
#     video_path = "uploaded_video.mp4"
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.read())

#     st.write("Processing the video...")

#     # Run processing steps
#     final_video_path = run_steps(video_path)

#     # Step 6: Download final video
#     st.write("Step 6: Download the final video.")
#     with open(final_video_path, "rb") as f:
#         st.download_button("Download Final Video", f, file_name="final_video.mp4", mime="video/mp4")

#     # Ask user if they want to process another video
#     st.write("Do you want to process another video?")
#     if st.button("Yes"):
#         st.experimental_rerun()  # Reload the app to process another video
#     else:
#         st.write("Thanks for using the app!")

import streamlit as st
from transformers import pipeline
import warnings
import librosa
from gtts import gTTS
from moviepy.editor import VideoFileClip
import ffmpeg
import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI client setup
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Suppress specific future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Whisper model once to avoid redundant reloading
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Set custom port
PORT = os.getenv("PORT", 8501)

# Function to transcribe audio with Whisper
def transcribe_audio(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=16000, duration=60)  # Load only the first 60 seconds
    transcription = whisper_model(audio)["text"]
    return transcription

# Function to extract audio from video
def extract_audio(video_file):
    audio_output = "extracted_audio.mp3"
    with VideoFileClip(video_file) as mp4_file:
        audio_only = mp4_file.audio
        audio_only.write_audiofile(audio_output)
    return audio_output

# Function to correct transcription using Azure LLM
def correct_transcription_azure(text):
    prompt =  """As a language correction assistant, your task is to improve the following transcription.\n
                1. Correct any grammatical mistakes.\n
                2. Remove filler words such as 'umm', 'hmm', and similar verbal pauses.\n
                3. Maintain the original meaning and intent of the transcription.\n
                4. Provide the corrected version without adding any new content or altering the original context.\n\n
                Original transcription:\n{text} """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    output_audio_file = "corrected.mp3"
    tts.save(output_audio_file)
    return output_audio_file

# Function to replace audio in a video using ffmpeg
def replace_audio(video_path, new_audio_path):
    output_video_path = "final_video.mp4"
    ffmpeg.input(video_path).output(ffmpeg.input(new_audio_path), output_video_path, vcodec="copy", acodec="aac").run(overwrite_output=True)
    return output_video_path

# Streamlit app layout
st.title("Welcome to Video Audio Processor")
st.write("Upload an MP4 video to process.")

# Steps function
def run_steps(video_path):
    step_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Step 1: Extract audio
    step_placeholder.write("Step 1: Extracting audio from video...")
    audio_path = extract_audio(video_path)
    progress_bar.progress(20)

    # Step 2: Concurrent tasks - transcribe audio and correct transcription
    with concurrent.futures.ThreadPoolExecutor() as executor:
        step_placeholder.write("Step 2: Transcribing audio and correcting transcription...")
        future_transcription = executor.submit(transcribe_audio, audio_path)
        transcription = future_transcription.result()

        # Correct transcription using Azure API in parallel
        future_correction = executor.submit(correct_transcription_azure, transcription)
        corrected_trans = future_correction.result()

    progress_bar.progress(60)

    # Step 3: Convert corrected transcription to speech
    step_placeholder.write("Step 3: Converting corrected transcription to speech...")
    tts_audio_path = text_to_speech(corrected_trans)
    progress_bar.progress(80)

    # Step 4: Replace audio in the video
    step_placeholder.write("Step 4: Replacing audio in video...")
    final_video_path = replace_audio(video_path, tts_audio_path)
    progress_bar.progress(100)

    return final_video_path

# File uploader
uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

if uploaded_file is not None:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing the video...")

    final_video_path = run_steps(video_path)

    st.write("Step 5: Download the final video.")
    with open(final_video_path, "rb") as f:
        st.download_button("Download Final Video", f, file_name="final_video.mp4", mime="video/mp4")

    if st.button("Process another video"):
        st.experimental_rerun()

# Run Streamlit on custom port
if __name__ == "__main__":
    st.run(port=int(PORT))


