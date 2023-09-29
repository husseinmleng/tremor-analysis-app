import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.signal as signal
from PIL import Image
import io

def plot_frequency_domain(file1_frequencies, file2_frequencies, sampling_rate=30, cutoff_frequency=5, window_size=5, max_magnitude_threshold=120):
    # Perform FFT on the tremor signal to obtain the frequency domain representation for File 1
    fft_values_file1 = np.fft.fft(file1_frequencies)
    frequencies_file1 = np.fft.fftfreq(len(fft_values_file1), d=1/sampling_rate)
    magnitude_file1 = np.abs(fft_values_file1)

    # Perform FFT on the tremor signal to obtain the frequency domain representation for File 2
    fft_values_file2 = np.fft.fft(file2_frequencies)
    frequencies_file2 = np.fft.fftfreq(len(fft_values_file2), d=1/sampling_rate)
    magnitude_file2 = np.abs(fft_values_file2)

    # Apply a low-pass filter to the magnitude spectrum for File 1
    cutoff_idx_file1 = int(cutoff_frequency * len(frequencies_file1) / (sampling_rate / 2))
    magnitude_file1[:cutoff_idx_file1] = signal.medfilt(magnitude_file1[:cutoff_idx_file1], kernel_size=3)

    # Apply a low-pass filter to the magnitude spectrum for File 2
    cutoff_idx_file2 = int(cutoff_frequency * len(frequencies_file2) / (sampling_rate / 2))
    magnitude_file2[:cutoff_idx_file2] = signal.medfilt(magnitude_file2[:cutoff_idx_file2], kernel_size=3)

    # Apply a moving average filter to the magnitude spectrum for File 1
    moving_avg_file1 = np.convolve(magnitude_file1, np.ones(window_size) / window_size, mode='same')

    # Apply a moving average filter to the magnitude spectrum for File 2
    moving_avg_file2 = np.convolve(magnitude_file2, np.ones(window_size) / window_size, mode='same')

    # Threshold the magnitude values that are greater than max_magnitude_threshold for File 1
    moving_avg_file1[moving_avg_file1 > max_magnitude_threshold] = max_magnitude_threshold

    # Threshold the magnitude values that are greater than max_magnitude_threshold for File 2
    moving_avg_file2[moving_avg_file2 > max_magnitude_threshold] = max_magnitude_threshold

    # Define the x and y values for plotting
    xPoints_file1 = frequencies_file1
    yPoints_file1 = moving_avg_file1

    xPoints_file2 = frequencies_file2
    yPoints_file2 = moving_avg_file2

    # Create the plot with specified range and styling
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xPoints_file1, yPoints_file1, label='File 1 Frequency Domain')
    ax.plot(xPoints_file2, yPoints_file2, label='File 2 Frequency Domain')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Frequency Domain Representation of Tremor Signal (Smoothed)')
    ax.set_xlim(0, 10)  # Set a reasonable upper frequency limit
    ax.set_ylim(0, max_magnitude_threshold * 1.1)

    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.185)

    # Convert the plot to an image and return it
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

def plot_tremor_signal(tremor_amplitudes1, tremor_amplitudes2, window_size=5):
    avg_amplitude1 = np.mean(tremor_amplitudes1)
    avg_amplitude2 = np.mean(tremor_amplitudes2)

    # Create time values based on the number of frames
    time1 = np.arange(len(tremor_amplitudes1))
    time2 = np.arange(len(tremor_amplitudes2))

    if avg_amplitude1 < 1:
        avg_amplitude1 = 0

    if avg_amplitude2 < 1:
        avg_amplitude2 = 0

    # Check if avg_amplitude is 0 for both files
    if avg_amplitude1 == 0 and avg_amplitude2 == 0:
        moving_avg_tremor1 = np.zeros_like(tremor_amplitudes1)
        moving_avg_tremor2 = np.zeros_like(tremor_amplitudes2)
    else:
        # Apply a moving average filter to the tremor signals
        moving_avg_tremor1 = np.convolve(tremor_amplitudes1, np.ones(window_size) / window_size, mode='same')
        moving_avg_tremor2 = np.convolve(tremor_amplitudes2, np.ones(window_size) / window_size, mode='same')

    # Plot the tremor signals for both files
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the tremor signals for both files
    ax1.plot(time1, moving_avg_tremor1, label='File 1 - Hand position', color='blue')
    ax1.plot(time2, moving_avg_tremor2, label='File 2 - Hand position', color='red')

    # Calculate the range and median amplitude for both files
    tremor_signal_cm1 = np.array(moving_avg_tremor1) * avg_amplitude1
    tremor_signal_cm2 = np.array(moving_avg_tremor2) * avg_amplitude2

    # Plot the range of tremor and median amplitude for both files
    ax1.axhline(y=max(tremor_signal_cm1), color='blue', linestyle='dotted', label='File 1 - Range of tremor')
    ax1.axhline(y=min(tremor_signal_cm1), color='blue', linestyle='dotted')
    ax1.axhline(y=max(tremor_signal_cm2), color='red', linestyle='dotted', label='File 2 - Range of tremor')
    ax1.axhline(y=min(tremor_signal_cm2), color='red', linestyle='dotted')

    ax1.axhline(y=np.median(tremor_signal_cm1) / 2, color='blue', linestyle='dashed', label='File 1 - Median amplitude')
    ax1.axhline(y=-np.median(tremor_signal_cm1) / 2, color='blue', linestyle='dashed')
    ax1.axhline(y=np.median(tremor_signal_cm2) / 2, color='red', linestyle='dashed', label='File 2 - Median amplitude')
    ax1.axhline(y=-np.median(tremor_signal_cm2) / 2, color='red', linestyle='dashed')

    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Tremor Amplitude (cm)')
    ax1.set_title('Waveform of Tremor with Measured Median Amplitude')
    ax1.legend()

    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.185)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

# Define a function to download plots as images
def download_plot_as_image(plot_img, filename,label="Download Plot"):
    b64_img = base64.b64decode(plot_img)
    st.download_button(
        label=label,
        data=b64_img,
        key=filename,
        file_name=filename,
    )


st.title("Tremor Signal Analysis")

uploaded_file1 = st.file_uploader("Upload the first Excel file", type=["xlsx", "xls"])
uploaded_file2 = st.file_uploader("Upload the second Excel file", type=["xlsx", "xls"])

large_plot_size = (16, 10)

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.title("Tremor Signal Analysis")

large_plot_size = (16, 10)

if uploaded_file1 is not None and uploaded_file2 is not None:
    df1 = pd.read_excel(uploaded_file1)
    df2 = pd.read_excel(uploaded_file2)

    required_columns = ["frame_number", "timestamp", "tremor_amplitude", "frequency_domain"]
    if not all(col in df1.columns for col in required_columns) or not all(col in df2.columns for col in required_columns):
        st.error("Both Excel files must contain columns: frame_number, timestamp, tremor_amplitude, and frequency_domain.")
    else:
        # Create a figure for Frequency Domain and Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        # Plot Frequency Domain for File 1 and File 2 in the same figure
        plot_img_frequency = plot_frequency_domain(df1["frequency_domain"].values, df2["frequency_domain"].values)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_frequency))), aspect='auto')
        ax.set_title("Frequency Domain Comparison")
        ax.legend(["File 1", "File 2"])


        st.pyplot(fig)

        # Create a figure for Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        # Plot Tremor Amplitude for File 1 and File 2 in the same figure
        plot_img_amplitudes = plot_tremor_signal(df1["tremor_amplitude"].values, df2["tremor_amplitude"].values)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_amplitudes))), aspect='auto')
        ax.set_title("Tremor Amplitude Comparison")
        ax.legend(["File 1", "File 2"])

        st.pyplot(fig)

        # Download plots as images
        download_plot_as_image(plot_img_frequency, "frequency_domain_plot.png","Download Frequency Domain Plot")
        download_plot_as_image(plot_img_amplitudes, "tremor_amplitude_plot.png","Download Tremor Amplitude Plot")