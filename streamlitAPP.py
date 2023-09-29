import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.signal as signal
from PIL import Image
import io

def plot_frequency_domain(tremor_amplitudes, sampling_rate=30, cutoff_frequency=5, window_size=5, max_magnitude_threshold=120):
    # Perform FFT on the tremor signal to obtain the frequency domain representation
    fft_values = np.fft.fft(tremor_amplitudes)
    frequencies = np.fft.fftfreq(len(fft_values), d=1/sampling_rate)
    magnitude = np.abs(fft_values)

    # Apply a low-pass filter to the magnitude spectrum
    cutoff_idx = int(cutoff_frequency * len(frequencies) / (sampling_rate / 2))
    magnitude[:cutoff_idx] = signal.medfilt(magnitude[:cutoff_idx], kernel_size=3)

    # Apply a moving average filter to the magnitude spectrum
    moving_avg = np.convolve(magnitude, np.ones(window_size) / window_size, mode='same')

    # Threshold the magnitude values that are greater than max_magnitude_threshold
    moving_avg[moving_avg > max_magnitude_threshold] = max_magnitude_threshold

    # Define the x and y values for plotting
    xPoints = frequencies
    yPoints = moving_avg

    # Create the plot with specified range and styling
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xPoints, yPoints, label='Frequency Domain')

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


def plot_tremor_signal(tremor_amplitudes, window_size=5):
    avg_amplitude = np.mean(tremor_amplitudes)

    # Create time values based on the number of frames
    time = np.arange(len(tremor_amplitudes))  # Define time outside of the conditional

    if avg_amplitude < 1:
        avg_amplitude = 0

    # Check if avg_amplitude is 0
    if avg_amplitude == 0:
        moving_avg_tremor = np.zeros_like(tremor_amplitudes)  # Create a signal of zeros
    else:
        # Create time values based on the number of frames
        time = np.arange(len(tremor_amplitudes))

        # Apply a moving average filter to the tremor signal
        moving_avg_tremor = np.convolve(tremor_amplitudes, np.ones(window_size) / window_size, mode='same')

    # Plot the tremor signal and related information
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the tremor signal
    ax1.plot(time, moving_avg_tremor, label='Hand position')

    # Calculate the range and median amplitude
    tremor_signal_cm = np.array(moving_avg_tremor) * avg_amplitude  # Assuming tremor amplitude is relative

    # Plot the range of tremor
    ax1.axhline(y=max(tremor_signal_cm), color='dimgrey', linestyle='dotted', label='Range of tremor')
    ax1.axhline(y=min(tremor_signal_cm), color='dimgrey', linestyle='dotted')

    # Plot the median amplitude
    ax1.axhline(y=np.median(tremor_signal_cm) / 2, color='dimgrey', linestyle='dashed', label='Median amplitude')
    ax1.axhline(y=-np.median(tremor_signal_cm) / 2, color='dimgrey', linestyle='dashed')

    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Tremor Amplitude (cm)')
    ax1.set_title('Waveform of a Tremor with a Measured Median Amplitude of %.2fcm' % np.median(tremor_signal_cm))

    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.185)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

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
        plot_img_frequency1 = plot_frequency_domain(df1["frequency_domain"].values)
        plot_img_frequency2 = plot_frequency_domain(df2["frequency_domain"].values)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_frequency1))), aspect='auto')
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_frequency2))), aspect='auto', alpha=0)
        ax.set_title("Frequency Domain Comparison")
        ax.legend(["File 1", "File 2"])


        st.pyplot(fig)

        # Create a figure for Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        # Plot Tremor Amplitude for File 1 and File 2 in the same figure
        plot_img_amplitudes1 = plot_tremor_signal(df1["tremor_amplitude"].values)
        plot_img_amplitudes2 = plot_tremor_signal(df2["tremor_amplitude"].values)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_amplitudes1))), aspect='auto')
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_amplitudes2))), aspect='auto', alpha=0.5)
        ax.set_title("Tremor Amplitude Comparison")
        ax.legend(["File 1", "File 2"])

        st.pyplot(fig)


        st.subheader("Download Plots as Images")
        st.markdown(
            f"Download Frequency Domain Plot (File 1 and File 2) as an image: [Download Plot](data:image/png;base64,{plot_img_frequency1})",
            unsafe_allow_html=True
        )
        st.markdown(
            f"Download Tremor Amplitude Plot (File 1 and File 2) as an image: [Download Plot](data:image/png;base64,{plot_img_amplitudes1})",
            unsafe_allow_html=True
        )