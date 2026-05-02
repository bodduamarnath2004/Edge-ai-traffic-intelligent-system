import sensor
import time
import audio
import pyb
import gc
from ulab import numpy as np
red_led   = pyb.LED(1)
green_led = pyb.LED(2)
red_led.off()
green_led.off()
SAMPLE_RATE  = 16000
CHUNK_SIZE   = 512
BYTE_CHUNK   = CHUNK_SIZE * 2
TARGET_BYTES = BYTE_CHUNK * 16
HALF         = CHUNK_SIZE // 2
SIREN_LO_BIN  = max(1,   int(500  * CHUNK_SIZE / SAMPLE_RATE))
SIREN_HI_BIN  = min(HALF-1, int(2000 * CHUNK_SIZE / SAMPLE_RATE))
NOISE_HI_BIN  = int(200  * CHUNK_SIZE / SAMPLE_RATE)
RMS_FLOOR             = 0.006
CONCENTRATION_THRESH  = 0.38
SNR_THRESH            = 4.0
WAIL_SWING_THRESH     = 4
MIN_REVERSALS         = 2
MIN_CHUNKS_FOR_WAIL   = 8
CONSECUTIVE_NEEDED    = 3
CONSECUTIVE_DECAY     = 2

peak_history      = []
MAX_HISTORY       = 30
consecutive_siren = 0

audio_buffer = bytearray()

def audio_callback(buf):
    global audio_buffer
    audio_buffer.extend(buf)
    if len(audio_buffer) > TARGET_BYTES * 2:
        audio_buffer = bytearray(audio_buffer[-TARGET_BYTES:])

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)

print("Siren Detector v3 - Balanced")
print("Siren band : 500-2000 Hz")
print("Consecutive: " + str(CONSECUTIVE_NEEDED) + " windows needed")

audio.init(channels=1, frequency=16000, gain_db=36, highpass=0.9883)
audio.start_streaming(audio_callback)
print("Listening")

def process_chunk(chunk_bytes):
    raw = np.frombuffer(chunk_bytes, dtype=np.int16)
    sig = np.array(raw, dtype=np.float) / 32768.0
    del raw
    gc.collect()
    rms = float(np.sqrt(np.mean(sig * sig)))
    fft_result = np.fft.fft(sig)
    del sig
    gc.collect()
    re = fft_result[0][:HALF]
    im = fft_result[1][:HALF]
    del fft_result
    gc.collect()
    power = re * re + im * im
    del re, im
    gc.collect()
    total_energy = float(np.sum(power))
    siren_band   = power[SIREN_LO_BIN:SIREN_HI_BIN]
    siren_energy = float(np.sum(siren_band))
    noise_energy = float(np.sum(power[1:NOISE_HI_BIN]))
    peak_local = 0
    peak_val   = 0.0
    for i in range(len(siren_band)):
        v = float(siren_band[i])
        if v > peak_val:
            peak_val   = v
            peak_local = i
    peak_bin = SIREN_LO_BIN + peak_local
    del siren_band, power
    gc.collect()
    concentration = siren_energy / total_energy if total_energy > 0.0 else 0.0
    snr           = siren_energy / (noise_energy + 1e-9)
    return rms, peak_bin, concentration, snr

def check_wailing(history):
    if len(history) < MIN_CHUNKS_FOR_WAIL:
        return False, 0, 0
    min_bin = history[0]
    max_bin = history[0]
    for b in history:
        if b < min_bin: min_bin = b
        if b > max_bin: max_bin = b
    swing = max_bin - min_bin
    reversals = 0
    for i in range(2, len(history)):
        d1 = history[i-1] - history[i-2]
        d2 = history[i]   - history[i-1]
        if d1 * d2 < 0:
            reversals += 1
    wailing = (swing >= WAIL_SWING_THRESH and reversals >= MIN_REVERSALS)
    return wailing, swing, reversals

def make_decision(avg_rms, avg_conc, avg_snr, wailing):
    if not wailing:
        return False, "NO_WAIL"
    if avg_rms < RMS_FLOOR:
        return False, "TOO_QUIET"
    conc_ok = avg_conc > CONCENTRATION_THRESH
    snr_ok  = avg_snr  > SNR_THRESH
    if not conc_ok and not snr_ok:
        return False, "POOR_SPECTRUM"
    return True, "SIREN"

last_process_time = time.ticks_ms()
PROCESS_INTERVAL  = 400
while True:
    now = time.ticks_ms()
    if (time.ticks_diff(now, last_process_time) >= PROCESS_INTERVAL
            and len(audio_buffer) >= TARGET_BYTES):
        last_process_time = now
        gc.collect()
        window_bytes = bytes(audio_buffer[-TARGET_BYTES:])
        audio_buffer = bytearray(audio_buffer[-256:])
        gc.collect()
        rms_total  = 0.0
        conc_total = 0.0
        snr_total  = 0.0
        count      = 0
        offset     = 0
        while offset + BYTE_CHUNK <= len(window_bytes):
            cb     = window_bytes[offset : offset + BYTE_CHUNK]
            offset += BYTE_CHUNK
            rms, peak_bin, conc, snr = process_chunk(cb)
            rms_total  += rms
            conc_total += conc
            snr_total  += snr
            count      += 1
            peak_history.append(peak_bin)
            if len(peak_history) > MAX_HISTORY:
                peak_history.pop(0)
        del window_bytes
        gc.collect()
        if count == 0:
            time.sleep_ms(50)
            continue
        avg_rms  = rms_total  / count
        avg_conc = conc_total / count
        avg_snr  = snr_total  / count
        wailing, swing, reversals = check_wailing(peak_history)
        is_siren, reason = make_decision(avg_rms, avg_conc,avg_snr, wailing)
        if is_siren:
            consecutive_siren = min(consecutive_siren + 1,CONSECUTIVE_NEEDED + 2)
        else:
            consecutive_siren = max(0,consecutive_siren - CONSECUTIVE_DECAY)
        confirmed = (consecutive_siren >= CONSECUTIVE_NEEDED)
        print("RMS="   + str(round(avg_rms,  4)) +
              " | Conc=" + str(round(avg_conc, 3)) +
              " | SNR="  + str(round(avg_snr,  1)) +
              " | Swing=" + str(swing) +
              " | Rev="   + str(reversals))
        print("Window=" + reason +
              " | Consec=" + str(consecutive_siren) +
              "/" + str(CONSECUTIVE_NEEDED))
        if confirmed:
            print(">>> SIREN DETECTED! <<<")
            red_led.on()
            green_led.off()
        else:
            print("    Normal  [" + reason + "]")
            red_led.off()
            green_led.on()
    time.sleep_ms(50)