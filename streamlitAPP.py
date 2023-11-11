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
    tremor_reduction = ((np.mean(file1_data["tremor_amplitude"].values) - np.mean(file2_data["tremor_amplitude"].values)) / np.mean(file1_data)) * 100
    print(tremor_reduction)
    
    # Calculate the increase in frequency amplitude
    # Assuming the increase in amplitude refers to the peak amplitude in the FFT
    fft_peak_file1 = max(np.abs(np.fft.fft(file1_data["frequency_domain"].values)))
    fft_peak_file2 = max(np.abs(np.fft.fft(file2_data["frequency_domain"].values)))
    freq_amp_increase = fft_peak_file2 - fft_peak_file1
    
    # Calculate the % increase in duration over a certain threshold (e.g., 0.5 Hz)
    # For simplicity, let's assume the threshold frequency is the mean frequency
    duration_file1 = len(file1_data["frequency_domain"].values) / len([f for f in file1_data["frequency_domain"].values if f > 0.5])
    duration_file2 = len(file2_data["frequency_domain"].values) / len([f for f in file2_data["frequency_domain"].values if f > 0.5])
    duration_increase = ((duration_file2 - duration_file1) / duration_file1) * 100
    
    return tremor_reduction, freq_amp_increase, duration_increase
 

def plot_frequency_domain(file1_frequencies, file2_frequencies, sampling_rate=30, cutoff_frequency=5, window_size=5, max_magnitude_threshold=120):
    # Perform FFT on the tremor signal to obtain the frequency domain representation for File 1
    fft_values_file1 = np.fft.fft(file1_frequencies["tremor_amplitude"].values)
    frequencies_file1 = np.fft.fftfreq(len(fft_values_file1), d=1/sampling_rate)
    magnitude_file1 = np.abs(fft_values_file1)

    # Perform FFT on the tremor signal to obtain the frequency domain representation for File 2
    fft_values_file2 = np.fft.fft(file2_frequencies["tremor_amplitude"].values)
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
    
    # Calculate the range and median amplitude for both files
    frequency_cm1 = np.array(moving_avg_file1) * np.mean(file1_frequencies["frequency_domain"].values)
    frequencies_cm2 = np.array(moving_avg_file2) * np.mean(file2_frequencies["frequency_domain"].values)

    _, freq_amp_increase, duration_increase = calculate_metrics(file1_frequencies, file2_frequencies)

    ax.annotate(f'Reduction in Frequency Amplitude: {np.abs(freq_amp_increase):.2f}%',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2))
    
    ax.plot(xPoints_file1, yPoints_file1, label='File 1 Frequency Domain', color='blue')
    ax.plot(xPoints_file2, yPoints_file2, label='File 2 Frequency Domain', color='red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Frequency Domain Representation of Tremor Signal (Smoothed)')
    ax.set_xlim(0, 10)  # Set a reasonable upper frequency limit
    ax.set_ylim(0, max_magnitude_threshold * 1.1)

    ax.legend()
    # Convert the plot to an image and return it
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_img

    
def plot_tremor_signal(tremor_amplitudes1, tremor_amplitudes2, window_size=5):
    avg_amplitude1 = np.mean(tremor_amplitudes1["tremor_amplitude"].values)
    avg_amplitude2 = np.mean(tremor_amplitudes2["tremor_amplitude"].values)

    # Create time values based on the number of frames
    time1 = np.arange(len(tremor_amplitudes1["tremor_amplitude"].values))
    time2 = np.arange(len(tremor_amplitudes2["tremor_amplitude"].values))

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
        moving_avg_tremor1 = np.convolve(tremor_amplitudes1["tremor_amplitude"].values, np.ones(window_size) / window_size, mode='same')
        moving_avg_tremor2 = np.convolve(tremor_amplitudes2["tremor_amplitude"].values, np.ones(window_size) / window_size, mode='same')

    # Plot the tremor signals for both files
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the tremor signals for both files
    ax1.plot(time1, moving_avg_tremor1, label='File 1 - Hand position', color='blue')
    ax1.plot(time2, moving_avg_tremor2, label='File 2 - Hand position', color='red')

    # Calculate the range and median amplitude for both files
    tremor_signal_cm1 = np.array(moving_avg_tremor1) * avg_amplitude1
    tremor_signal_cm2 = np.array(moving_avg_tremor2) * avg_amplitude2

    tremor_reduction, _, _ = calculate_metrics(tremor_amplitudes1, tremor_amplitudes2)

    # Annotate the plot with the tremor reduction metric
    ax1.annotate(f'Reduction in Tremor: {np.abs(np.mean(tremor_reduction)):.2f}%', 
                 xy=(0.5, 0.95), xycoords='axes fraction', 
                 ha='center', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2))


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


def plot_spectrogram(file1_frequencies, file2_frequencies, sampling_rate=5, cutoff_frequency=1, window_size=5, max_magnitude_threshold=120):
    # Perform FFT on the tremor signal to obtain the frequency domain representation for File 1 and File 2
    fft_values_file1 = np.fft.fft(file1_frequencies["tremor_amplitude"])
    fft_values_file2 = np.fft.fft(file2_frequencies["tremor_amplitude"])

    # Calculate the frequency bins for the FFT output
    frequencies = np.fft.fftfreq(len(fft_values_file1), d=1/sampling_rate)
    
    # Apply a low-pass filter to the magnitude spectrum for File 1 and File 2
    cutoff_idx = int(cutoff_frequency * len(frequencies) / (sampling_rate / 2))
    magnitude_file1 = np.abs(fft_values_file1)
    magnitude_file1[:cutoff_idx] = signal.medfilt(magnitude_file1[:cutoff_idx], kernel_size=3)
    magnitude_file2 = np.abs(fft_values_file2)
    magnitude_file2[:cutoff_idx] = signal.medfilt(magnitude_file2[:cutoff_idx], kernel_size=3)

    # Apply a moving average filter to the magnitude spectrum for File 1 and File 2
    moving_avg_file1 = np.convolve(magnitude_file1, np.ones(window_size) / window_size, mode='same')
    moving_avg_file2 = np.convolve(magnitude_file2, np.ones(window_size) / window_size, mode='same')

    # Threshold the magnitude values for both files
    moving_avg_file1 = np.clip(moving_avg_file1, None, max_magnitude_threshold)
    moving_avg_file2 = np.clip(moving_avg_file2, None, max_magnitude_threshold)

    # Calculate the Short-Time Fourier Transform for both tremor signals
    nperseg = int(window_size * sampling_rate)
    frequencies1, times1, Sxx1 = signal.spectrogram(moving_avg_file1, fs=sampling_rate, nperseg=nperseg)
    frequencies2, times2, Sxx2 = signal.spectrogram(moving_avg_file2, fs=sampling_rate, nperseg=nperseg)

    # Apply a low-pass filter to each segment's FFT magnitudes for both files
    cutoff_idx = int(cutoff_frequency * nperseg / (sampling_rate / 2))
    Sxx1[:, :cutoff_idx] = signal.medfilt(Sxx1[:, :cutoff_idx], kernel_size=(1, 3))
    Sxx2[:, :cutoff_idx] = signal.medfilt(Sxx2[:, :cutoff_idx], kernel_size=(1, 3))

    # Apply a moving average filter to the magnitude spectrum for both files
    for i in range(Sxx1.shape[1]):
        Sxx1[:, i] = np.convolve(Sxx1[:, i], np.ones(window_size) / window_size, mode='same')
        Sxx2[:, i] = np.convolve(Sxx2[:, i], np.ones(window_size) / window_size, mode='same')

    # Threshold the magnitude values that are greater than max_magnitude_threshold for both files
    Sxx1[Sxx1 > max_magnitude_threshold] = max_magnitude_threshold
    Sxx2[Sxx2 > max_magnitude_threshold] = max_magnitude_threshold
    # Define fixed limits for the color scale based on intensity in dB
    # Calculate the intensity values for both files
    intensity_values_file1 = 10 * np.log10(Sxx1)
    intensity_values_file2 = 10 * np.log10(Sxx2)

    # Find the global minimum and maximum intensity values after ensuring there are no negative or zero values in the Sxx arrays
    global_intensity_min = np.min([intensity_values_file1[intensity_values_file1 > -np.inf], intensity_values_file2[intensity_values_file2 > -np.inf]])
    global_intensity_max = np.max([intensity_values_file1, intensity_values_file2])

    # Create subplots with a larger figsize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20)) # Adjust figsize here as needed
    # Additionally, adjust the subplots to fill the figure area
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


   
    # Plot Spectrogram for File 1
    fig1, ax1 = plt.subplots(figsize=(20, 12))
    cmap = plt.get_cmap('magma')
    spec1 = ax1.pcolormesh(times1, (frequencies1), 10 * np.log10(Sxx1), shading='gouraud', cmap=cmap, vmin=0, vmax=15)
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time')
    ax1.set_title('Spectrogram for File 1')
    fig1.colorbar(spec1, ax=ax1, orientation='vertical', label='Intensity')
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    buf1.seek(0)
    plot_img1 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)

    # Plot Spectrogram for File 2
    fig2, ax2 = plt.subplots(figsize=(20, 12))
    spec2 = ax2.pcolormesh(times2, (frequencies2), 10 * np.log10(Sxx2), shading='gouraud', cmap=cmap, vmin=0, vmax=15)
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time')
    ax2.set_title('Spectrogram for File 2')
    fig2.colorbar(spec2, ax=ax2, orientation='vertical', label='Intensity')
    buf2 = BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    buf2.seek(0)
    plot_img2 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)


    return plot_img1, plot_img2


st.title("Tremor Signal Analysis")

uploaded_file1 = st.file_uploader("Upload the first Excel file", type=["xlsx", "xls"], key="1")
uploaded_file2 = st.file_uploader("Upload the second Excel file", type=["xlsx", "xls"], key="2")

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
        plot_img_frequency = plot_frequency_domain(df1, df2)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_frequency))), aspect='auto')
        ax.set_title("Frequency Domain Comparison")
        ax.legend(["File 1", "File 2"])


        st.pyplot(fig)

        # Create a figure for Tremor Amplitude plots
        fig, ax = plt.subplots(1, 1, figsize=large_plot_size)

        # Plot Tremor Amplitude for File 1 and File 2 in the same figure
        plot_img_amplitudes = plot_tremor_signal(df1, df2)
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(plot_img_amplitudes))), aspect='auto')
        ax.set_title("Tremor Amplitude Comparison")
        ax.legend(["File 1", "File 2"])

        st.pyplot(fig)

        plot_img1, plot_img2 = plot_spectrogram(df1, df2)

        # Display the spectrogram for File 1
        st.subheader("Spectrogram for File 1")
        st.image(Image.open(BytesIO(base64.b64decode(plot_img1))), caption="Spectrogram for File 1", use_column_width=True)

        # Display the spectrogram for File 2 
        st.subheader("Spectrogram for File 2")
        st.image(Image.open(BytesIO(base64.b64decode(plot_img2))), caption="Spectrogram for File 2", use_column_width=True)

        # Download plots as images
        download_plot_as_image(plot_img_frequency, "frequency_domain_plot.png","Download Frequency Domain Plot")
        download_plot_as_image(plot_img_amplitudes, "tremor_amplitude_plot.png","Download Tremor Amplitude Plot")
        download_plot_as_image(plot_img1, "spectrogram_plot1.png","Download Spectrogram Plot 1")
        download_plot_as_image(plot_img2, "spectrogram_plot2.png","Download Spectrogram Plot 2")