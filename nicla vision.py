import sensor, time, network, socket
import imu  
SSID = "XXXXXXXXX"
KEY  = "XXXXXXXXX"
PORT = 8080
sensor.reset()
sensor.set_framesize(sensor.QVGA)
sensor.set_pixformat(sensor.RGB565)
vx, vy = 0.0, 0.0
last_time = time.ticks_ms()
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, KEY)
print("Connecting")
while not wlan.isconnected():
    time.sleep_ms(500)
print("Connected! IP:", wlan.ifconfig()[0])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
s.bind(('', PORT))
s.listen(1)
def start_streaming(s):
    global vx, vy, last_time
    client, addr = s.accept()
    client.settimeout(5.0)
    client.sendall("HTTP/1.1 200 OK\r\n"
                   "Content-Type: multipart/x-mixed-replace;boundary=openmv\r\n"
                   "Cache-Control: no-cache\r\n\r\n")
    while True:
        current_time = time.ticks_ms()
        dt = time.ticks_diff(current_time, last_time) / 1000.0
        last_time = current_time
        ax_mg, ay_mg, az_mg = imu.acceleration_mg()
        ax = ax_mg * 0.00981
        ay = ay_mg * 0.00981
        if abs(ax) < 0.2: ax = 0
        if abs(ay) < 0.2: ay = 0
        vx += ax * dt
        vy += ay * dt
        vx *= 0.95 
        vy *= 0.95
        total_speed = (vx**2 + vy**2)**0.5
        frame = sensor.snapshot()
        cframe = frame.to_jpeg(quality=35)
        header = ("\r\n--openmv\r\n"
                  "Content-Type: image/jpeg\r\n"
                  "Content-Length: " + str(cframe.size()) + "\r\n"
                  "X-Speed: " + str(total_speed) + "\r\n\r\n")
        client.sendall(header)
        client.sendall(cframe)

while True:
    try:
        start_streaming(s)
    except OSError as e:
        print("Connection error:", e)
        vx, vy = 0.0, 0.0 
        time.sleep_ms(500)