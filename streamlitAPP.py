import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.signal as signal
from PIL import Image
import io


def calculate_metrics(file1_data, file2_data):
    # Calculate the percentage reduction in tremor amplitude
    tremor_reduction = ((np.mean(file1_data["tremor_amplitude"].values) - np.mean(
        file2_data["tremor_amplitude"].values)) / np.mean(file1_data)) * 100
    print(tremor_reduction)

    # Calculate the increase in frequency amplitude
    # Assuming the increase in amplitude refers to the peak amplitude in the FFT
    fft_peak_file1 = max(np.abs(np.fft.fft(file1_data["frequency_domain"].values)))
    fft_peak_file2 = max(np.abs(np.fft.fft(file2_data["frequency_domain"].values)))
    freq_amp_increase = fft_peak_file2 - fft_peak_file1

    # Calculate the % increase in duration over a certain threshold (e.g., 0.5 Hz)
    # For simplicity, let's assume the threshold frequency is the mean frequency
    duration_file1 = len(file1_data["frequency_domain"].values) / len(
        [f for f in file1_data["frequency_domain"].values if f > 0.5])
    duration_file2 = len(file2_data["frequency_domain"].values) / len(
        [f for f in file2_data["frequency_domain"].values if f > 0.5])
    duration_increase = ((duration_file2 - duration_file1) / duration_file1) * 100

    return tremor_reduction, freq_amp_increase, duration_increase


def plot_frequency_domain(df1, df2, df3, df4, df5, metrics, sampling_rate=30, cutoff_frequency=5, window_size=5,
                          max_magnitude_threshold=120):
    # Perform FFT and apply filters for each data frame
    # This process is repeated for each data frame (df1, df2, df3, df4, df5)
    fft_values = [np.fft.fft(df["tremor_amplitude"].values) for df in [df1, df2, df3, df4, df5]]
    frequencies = [np.fft.fftfreq(len(values), d=1 / sampling_rate) for values in fft_values]
    magnitudes = [np.abs(values) for values in fft_values]

    # Applying low-pass filter and moving average for each magnitude
    for idx, magnitude in enumerate(magnitudes):
        cutoff_idx = int(cutoff_frequency * len(frequencies[idx]) / (sampling_rate / 2))
        magnitude[:cutoff_idx] = signal.medfilt(magnitude[:cutoff_idx], kernel_size=3)
        magnitudes[idx] = np.convolve(magnitude, np.ones(window_size) / window_size, mode='same')
        magnitudes[idx] = np.clip(magnitudes[idx], None, max_magnitude_threshold)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Different colors for each data frame
    labels = ['File 1', 'File 2', 'File 3', 'File 4', 'File 5']  # Labels for each data frame

    # Display the calculated metrics on the plot
    tremor_reduction, freq_amp_increase, duration_increase = metrics

    ax.annotate(f'Reduction in Frequency Amplitude: {np.abs(freq_amp_increase):.2f}%',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2))

    for freq, mag, color, label in zip(frequencies, magnitudes, colors, labels):
        ax.plot(freq, mag, label=label, color=color)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Frequency Domain Representation of Tremor Signal (Smoothed)')
    ax.set_xlim(0, 10)  # Adjust frequency limits as needed
    ax.set_ylim(0, max_magnitude_threshold * 1.1)
    ax.legend()

    # Convert the plot to an image and return it
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

def plot_mean_amplitude(df_list):
    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and labels for each data frame
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = ['File 1', 'File 2', 'File 3', 'File 4', 'File 5']

    # Calculate mean amplitudes for each data frame
    mean_amplitudes = [np.mean(df["tremor_amplitude"].values) for df in df_list]

    # Days for the x-axis
    days = range(1, len(df_list) + 1)

    # Plot each mean amplitude for each day with a line connecting them
    ax.plot(days, mean_amplitudes, marker='o', linestyle='-', color='red')

    # Set plot labels and title
    ax.set_xlabel('File')
    ax.set_ylabel('Mean Amplitude')
    ax.set_title('Tremor amplitude over time')

    # Adding labels for each point
    for i, txt in enumerate(labels):
        ax.annotate(txt, (days[i], mean_amplitudes[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Convert the plot to an image and return it
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

def plot_tremor_signal(df1, df2, df3, df4, df5, metrics, window_size=5):
    # Calculate average amplitudes and times for each data frame
    avg_amplitudes = [np.mean(df["tremor_amplitude"].values) for df in [df1, df2, df3, df4, df5]]
    times = [np.arange(len(df["tremor_amplitude"].values)) for df in [df1, df2, df3, df4, df5]]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and labels for each data frame
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = ['File 1', 'File 2', 'File 3', 'File 4', 'File 5']
    # Display the calculated metrics on the plot
    tremor_reduction, freq_amp_increase, duration_increase = metrics
    # Annotate the plot with the tremor reduction metric
    ax.annotate(f'Reduction in Tremor: {np.abs(np.mean(tremor_reduction)):.2f}%',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2))

    # Plot each data frame
    for df, avg_amplitude, time, color, label in zip([df1, df2, df3, df4, df5], avg_amplitudes, times, colors, labels):

        moving_avg_tremor = np.convolve(df["tremor_amplitude"].values, np.ones(window_size) / window_size,
                                            mode='same')
        ax.plot(time, moving_avg_tremor, label=label, color=color)

    # Set plot labels and title
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Tremor Amplitude (cm)')
    ax.set_title('Waveform of Tremor with Measured Median Amplitude')
    ax.legend()

    # Save the plot to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img


# Define a function to download plots as images
def download_plot_as_image(plot_img, filename, label="Download Plot"):
    b64_img = base64.b64decode(plot_img)
    st.download_button(
        label=label,
        data=b64_img,
        key=filename,
        file_name=filename,
    )

st.title("Tremor Signal Analysis")

uploaded_file1 = st.file_uploader("Upload the 1st Excel file", type=["xlsx", "xls"], key="1")
uploaded_file2 = st.file_uploader("Upload the 2nd Excel file", type=["xlsx", "xls"], key="2")
uploaded_file3 = st.file_uploader("Upload the 3rd Excel file", type=["xlsx", "xls"], key="3")
uploaded_file4 = st.file_uploader("Upload the 4th Excel file", type=["xlsx", "xls"], key="4")
uploaded_file5 = st.file_uploader("Upload the 5th Excel file", type=["xlsx", "xls"], key="5")

large_plot_size = (16, 10)

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.title("Tremor Signal Analysis")

large_plot_size = (16, 10)

if uploaded_file1 is not None and uploaded_file2 is not None:
    df1 = pd.read_excel(uploaded_file1)
    df2 = pd.read_excel(uploaded_file2)
    df3 = pd.read_excel(uploaded_file3)
    df4 = pd.read_excel(uploaded_file4)
    df5 = pd.read_excel(uploaded_file5)

    required_columns = ["frame_number", "timestamp", "tremor_amplitude", "frequency_domain"]
    if not all(col in df1.columns for col in required_columns) or not all(
            col in df2.columns for col in required_columns):
        st.error(
            "Both Excel files must contain columns: frame_number, timestamp, tremor_amplitude, and frequency_domain.")
    else:
        # Create a figure for Frequency Domain and Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        metrics = calculate_metrics(df1, df5)

        # Plot Frequency Domain for File 1 and File 2 in the same figure
        plot_img_frequency = plot_frequency_domain(df1, df2, df3, df4, df5, metrics)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_frequency))), aspect='auto')
        ax.set_title("Frequency Domain Comparison")
        ax.legend(["File 1", "File 2", "File 3", "File 4", "File 5"])

        st.pyplot(fig)

        # Create a figure for Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        # Plot Tremor Amplitude for File 1 and File 2 in the same figure
        plot_img_amplitudes = plot_tremor_signal(df1, df2, df3, df4, df5, metrics)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_amplitudes))), aspect='auto')
        ax.set_title("Tremor Amplitude Comparison")
        ax.legend(["File 1", "File 2", "File 3", "File 4", "File 5"])
        st.pyplot(fig)

        # Create a figure for Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)
        #plot mean amplitude
        plot_img_mean_amplitude = plot_mean_amplitude([df1, df2, df3, df4, df5])
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_mean_amplitude))), aspect='auto')
        ax.set_title("Mean Amplitude Comparison")
        ax.legend(["File 1", "File 2", "File 3", "File 4", "File 5"])

        st.pyplot(fig)

        download_plot_as_image(plot_img_frequency, "frequency_domain_plot.png", "Download Frequency Domain Plot")
        download_plot_as_image(plot_img_amplitudes, "tremor_amplitude_plot.png", "Download Tremor Amplitude Plot")