import cv2
import numpy as np
import time
import collections
import urllib.request
import threading
import queue
import os
import psutil
import tracemalloc
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class EdgeAIMetrics:
    def __init__(self, cam_name: str):
        self.cam_name            = cam_name
        self._lock               = threading.Lock()
        self.inference_times_ms  = collections.deque(maxlen=100)  
        self.frame_latencies_ms  = collections.deque(maxlen=100)  
        self.frames_processed    = 0
        self.start_time          = time.time()
        self._proc               = psutil.Process(os.getpid())
        self.peak_rss_mb         = 0.0        
        self.model_footprint_mb  = 0.0          
        self.cpu_samples         = collections.deque(maxlen=50)
        tracemalloc.start()
    def record_inference(self, elapsed_ms: float):
        with self._lock:
            self.inference_times_ms.append(elapsed_ms)

    def record_frame_latency(self, elapsed_ms: float):
        with self._lock:
            self.frame_latencies_ms.append(elapsed_ms)
            self.frames_processed += 1
            rss = self._proc.memory_info().rss / 1e6
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            self.cpu_samples.append(self._proc.cpu_percent(interval=None))

    def set_model_footprint(self, model_path: str):
        """Approximate model disk footprint in MB."""
        try:
            self.model_footprint_mb = os.path.getsize(model_path) / 1e6
        except OSError:
            self.model_footprint_mb = 0.0
    def snapshot(self) -> dict:
        with self._lock:
            inf  = list(self.inference_times_ms)
            lat  = list(self.frame_latencies_ms)
            cpu  = list(self.cpu_samples)
            elapsed = max(time.time() - self.start_time, 1e-3)
            cur, peak_heap = tracemalloc.get_traced_memory()
            return {
                'avg_inference_ms'   : float(np.mean(inf))  if inf  else 0.0,
                'p95_inference_ms'   : float(np.percentile(inf, 95)) if inf else 0.0,
                'avg_frame_lat_ms'   : float(np.mean(lat))  if lat  else 0.0,
                'p95_frame_lat_ms'   : float(np.percentile(lat, 95)) if lat else 0.0,
                'throughput_fps'     : self.frames_processed / elapsed,
                'frames_processed'   : self.frames_processed,
                'model_footprint_mb' : self.model_footprint_mb,
                'peak_rss_mb'        : self.peak_rss_mb,
                'cur_heap_mb'        : cur  / 1e6,
                'peak_heap_mb'       : peak_heap / 1e6,
                'avg_cpu_pct'        : float(np.mean(cpu)) if cpu else 0.0,
                'peak_cpu_pct'       : float(np.max(cpu))  if cpu else 0.0,
            }

    def print_report(self):
        s = self.snapshot()
        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  Edge AI Metrics — {self.cam_name}")
        print(sep)
        print(f"  [Inference Latency]")
        print(f"    Avg  YOLO inference : {s['avg_inference_ms']:>8.2f} ms")
        print(f"    P95  YOLO inference : {s['p95_inference_ms']:>8.2f} ms")
        print(f"    Avg  frame latency  : {s['avg_frame_lat_ms']:>8.2f} ms")
        print(f"    P95  frame latency  : {s['p95_frame_lat_ms']:>8.2f} ms")
        print(f"  [Throughput]")
        print(f"    Processed frames    : {s['frames_processed']:>8d}")
        print(f"    Throughput          : {s['throughput_fps']:>8.2f} FPS")
        print(f"  [Memory Footprint]")
        print(f"    Model file size     : {s['model_footprint_mb']:>8.2f} MB")
        print(f"    Peak RSS (proc)     : {s['peak_rss_mb']:>8.2f} MB")
        print(f"    Current heap (Py)   : {s['cur_heap_mb']:>8.2f} MB")
        print(f"    Peak heap (Py)      : {s['peak_heap_mb']:>8.2f} MB")
        print(f"  [CPU Usage]")
        print(f"    Avg CPU             : {s['avg_cpu_pct']:>8.2f} %")
        print(f"    Peak CPU            : {s['peak_cpu_pct']:>8.2f} %")
        print(sep)

NICLA_CAMERAS = [
    {"id": 0, "ip": "XXXXXXXXXXX", "port": 8080, "name": "Camera 1", "role": "FRONT"},  #change IP1
    {"id": 1, "ip": "XXXXXXXXXXX", "port": 8080, "name": "Camera 2", "role": "REAR"},   #change IP2
]
CFG = {
    'yolo_model'            : 'yolo11n_vehicle_qat_final.pt',  
    'yolo_imgsz'            : 320,
    'conf_thresh'           : 0.40,
    'iou_thresh'            : 0.45,
    'target_classes'        : [2, 3, 5, 7],  
    'yolo_every_n'          : 3,
    'max_age'               : 10,
    'min_hits'              : 2,
    'iou_threshold'         : 0.25,
    'use_egomotion'         : True,
    'ego_corners'           : 150,
    'ego_subtract_factor'   : 0.2,
    'lk_winSize'            : (15, 15),
    'lk_maxLevel'           : 2,
    'scale_mpp'             : 0.20,
    'min_flow_px'           : 0.1,
    'smoothing_n'           : 6,
    'assumed_fps'           : 30.0,
    'display_width'         : 1280,
    'tailgate_box_h_ratio'  : 0.35,   
    'braking_dh_thresh'     : 3.0,    
    'congestion_count'      : 5,     
    'congestion_speed'      : 20.0, 
    'risk_score_high'       : 70,
    'risk_score_med'        : 40,
    'speed_variance_high'   : 200.0,
    'speed_variance_med'    : 80.0,
}
print("Config loaded | Dual camera mode")
class NiclaStream:
    def __init__(self, url):
        self.url          = url
        self._lock        = threading.Lock()
        self._latest      = None
        self._has_frame   = threading.Event()
        self._stop        = threading.Event()
        self._speed_lock  = threading.Lock()
        self.ego_speed_ms = 0.0 
        self._thread      = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _connect(self):
        print(f"Connecting to {self.url} ...")
        while not self._stop.is_set():
            try:
                s = urllib.request.urlopen(self.url, timeout=10)
                print(f"Connected to {self.url}!")
                return s
            except Exception as e:
                print(f"Connection failed ({self.url}): {e}. Retrying in 2s...")
                time.sleep(2)
        return None

    def _drain(self):
        buf    = bytes()
        stream = self._connect()
        if stream is None:
            return
        while not self._stop.is_set():
            try:
                chunk = stream.read(4096)
                if not chunk:
                    print(f"Stream ended ({self.url}), reconnecting...")
                    stream = self._connect()
                    buf = bytes()
                    continue
                buf += chunk
                while True:
                    hdr_end = buf.find(b'\r\n\r\n')
                    if hdr_end != -1:
                        hdr_block = buf[:hdr_end].decode('utf-8', errors='ignore')
                        for line in hdr_block.splitlines():
                            if line.lower().startswith('x-speed:'):
                                try:
                                    spd_val = float(line.split(':', 1)[1].strip())
                                    with self._speed_lock:
                                        self.ego_speed_ms = spd_val
                                except ValueError:
                                    pass

                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9', start)
                    if start == -1 or end == -1 or end <= start:
                        break
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]        
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is not None:
                        with self._lock:
                            self._latest = img 
                        self._has_frame.set()
            except Exception as e:
                print(f"Stream error ({self.url}): {e}. Reconnecting...")
                try: stream.close()
                except: pass
                stream = self._connect()
                buf = bytes()

    def read(self):
        self._has_frame.wait(timeout=5.0)
        with self._lock:
            frame = self._latest
        if frame is None:
            return False, None
        return True, frame.copy()

    def get_ego_speed_ms(self):
        with self._speed_lock:
            return self.ego_speed_ms

    def stop(self):
        self._stop.set()

def iou_batch(bb_test, bb_gt):
    bb_gt   = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w   = np.maximum(0., xx2 - xx1)
    h   = np.maximum(0., yy2 - yy1)
    inter = w * h
    a1  = (bb_test[..., 2]-bb_test[..., 0]) * (bb_test[..., 3]-bb_test[..., 1])
    a2  = (bb_gt[...,  2]-bb_gt[...,  0]) * (bb_gt[...,  3]-bb_gt[...,  1])
    return inter / (a1 + a2 - inter + 1e-6)

def convert_bbox_to_z(bbox):
    w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.;    y = bbox[1]+h/2.
    return np.array([x, y, w*h, w/float(h+1e-6)]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w   = np.sqrt(abs(x[2]) * abs(x[3]))
    h   = abs(x[2]) / (w + 1e-6)
    box = [x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]
    if score is None:
        return np.array(box).reshape((1, 4))
    return np.array([*box, score]).reshape((1, 5))

class KalmanBoxTracker:
    def __init__(self, bbox, cls_id, id_counter_ref):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                               [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                               [0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
                               [0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=np.float32)
        self.kf.R[2:, 2:] *= 10.; self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.; self.kf.Q[-1,-1] *= 0.01; self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = id_counter_ref[0]; id_counter_ref[0] += 1
        self.cls_id = cls_id
        self.hits = self.hit_streak = self.age = self.time_since_update = 0
        self.history_centers = collections.deque(maxlen=CFG['smoothing_n'])
        self.speed_kmh = 0.0

    def update(self, bbox):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0: self.kf.x[6] = 0.
        self.kf.predict(); self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def update_speed(self, ego_px, fps):
        box   = self.get_state()[0]
        cx    = (box[0]+box[2])/2.; cy = (box[1]+box[3])/2.
        box_h = max(1., box[3]-box[1])
        self.history_centers.append((cx, cy, box_h))
        if len(self.history_centers) < 2: return
        pts       = np.array(self.history_centers)
        dx        = np.diff(pts[:,0]); dy = np.diff(pts[:,1])
        disp      = np.sqrt(dx**2+dy**2)
        raw_flow  = float(np.mean(disp))
        rel_flow  = max(0., raw_flow - ego_px*CFG['ego_subtract_factor'])
        if rel_flow < CFG['min_flow_px']: rel_flow = 0.
        avg_box_h = float(np.mean(pts[:,2]))
        depth_scale = 100./(avg_box_h+1e-3)
        self.speed_kmh = rel_flow * CFG['scale_mpp'] * depth_scale * fps * 3.6

class SORTTracker:
    def __init__(self):
        self.trackers    = []
        self._id_counter = [0]

    def update(self, detections, cls_ids):
        trks   = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]; trks[t] = pos[:4]
            if np.any(np.isnan(pos)): to_del.append(t)
        for t in reversed(to_del): self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        matched, unmatched_dets, _ = self._associate(detections, trks)
        for m in matched: self.trackers[m[1]].update(detections[m[0]])
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i], cls_ids[i], self._id_counter))
        self.trackers = [t for t in self.trackers if t.time_since_update <= CFG['max_age']]
        return self.trackers

    def _associate(self, detections, trackers):
        if len(trackers) == 0: return [], list(range(len(detections))), []
        if len(detections) == 0: return [], [], list(range(len(trackers)))
        iou_mat = iou_batch(detections, trackers)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_indices  = np.stack([row_ind, col_ind], axis=1)
        unmatched_dets   = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
        unmatched_trks   = [t for t in range(len(trackers))   if t not in matched_indices[:,1]]
        matched          = [m for m in matched_indices if iou_mat[m[0],m[1]] >= CFG['iou_threshold']]
        unmatched_dets  += [m[0] for m in matched_indices if iou_mat[m[0],m[1]] < CFG['iou_threshold']]
        return matched, unmatched_dets, unmatched_trks

def estimate_ego_motion(prev_gray, curr_gray):
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=CFG['ego_corners'],
                                  qualityLevel=0.02, minDistance=10, blockSize=5)
    if pts is None or len(pts) < 6: return 0.0
    lk_params = dict(winSize=CFG['lk_winSize'], maxLevel=CFG['lk_maxLevel'],
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk_params)
    if next_pts is None or status is None: return 0.0
    good_prev = pts[status==1]; good_next = next_pts[status==1]
    if len(good_prev) < 4: return 0.0
    M, _ = cv2.estimateAffinePartial2D(good_prev, good_next,
                                        method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None: return 0.0
    return float(np.clip(np.sqrt(M[0,2]**2+M[1,2]**2), 0., 50.))

LABEL_MAP = {2:'car', 3:'motorbike', 5:'bus', 7:'truck'}
COLOR_MAP  = {2:(50,200,255), 3:(50,255,150), 5:(255,120,50), 7:(180,50,255)}

def speed_color(kmh):
    if kmh < 30:  return (80, 255, 80)
    elif kmh < 60: return (0, 220, 255)
    else:          return (50, 50, 255)

def draw_overlay(frame, trackers, ego_px, fps_actual, frame_idx, cam_name,
                 ego_speed_kmh=0.0):   
    h_frame, w_frame = frame.shape[:2]
    for trk in trackers:
        if trk.hit_streak < CFG['min_hits']: continue
        box = trk.get_state()[0]
        x1  = max(0, int(box[0]))
        y1  = max(0, int(box[1]))
        x2  = min(w_frame-1, int(box[2]))
        y2  = min(h_frame-1, int(box[3]))
        if x2 <= x1 or y2 <= y1: continue
        abs_speed = max(0.0, trk.speed_kmh)              
        rel_speed = max(0.0, trk.speed_kmh - ego_speed_kmh)   
        if ego_speed_kmh < 5.0:         
            rel_speed = abs_speed
        color   = COLOR_MAP.get(trk.cls_id, (200,200,200))
        spd_col = speed_color(rel_speed)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        cl, tc = 22, 5
        for px,py,dx,dy in [(x1,y1,+1,+1),(x2,y1,-1,+1),(x1,y2,+1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px,py), (px+dx*cl,py), color, tc)
            cv2.line(frame, (px,py), (px,py+dy*cl), color, tc)
        speed_txt = f'{rel_speed:.0f}/{abs_speed:.0f} km/h'
        font = cv2.FONT_HERSHEY_DUPLEX
        fs, ft = 0.62, 1
        (sw,sh),bl = cv2.getTextSize(speed_txt, font, fs, ft)
        pad = 6
        if y1 - sh - bl - pad*2 > 0:
            bx1,by1,bx2,by2 = x1, y1-sh-bl-pad*2, x1+sw+pad*2, y1
        else:
            bx1,by1,bx2,by2 = x1, y1, x1+sw+pad*2, y1+sh+bl+pad*2
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (15,15,15), -1)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), spd_col, 2)
        tx, ty = bx1+pad, by2-bl-pad//2
        cv2.putText(frame, speed_txt, (tx+1,ty+1), font, fs, (0,0,0),       ft+2, cv2.LINE_AA)
        cv2.putText(frame, speed_txt, (tx,  ty),   font, fs, (255,255,255),  ft,   cv2.LINE_AA)
        bar_w = int((x2-x1)*min(rel_speed/100., 1.0))
        cv2.rectangle(frame, (x1,y2+4), (x1+bar_w,y2+9), spd_col, -1)
        cv2.rectangle(frame, (x1,y2+4), (x2,      y2+9), color,    1)
    n_active = sum(1 for t in trackers if t.hit_streak >= CFG['min_hits'])
    hud = (f'[{cam_name}] Frame {frame_idx} (LIVE) | FPS {fps_actual:.1f} | '
           f'ego {ego_px:.1f}px | EGO {ego_speed_kmh:.1f} km/h | tracks {n_active}')
    cv2.rectangle(frame, (0,h_frame-30), (w_frame,h_frame), (0,0,0), -1)
    cv2.putText(frame, hud, (10,h_frame-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200,200,200), 1, cv2.LINE_AA)
    return frame

def compute_analytics(trackers, ego_px, fps, frame_h, frame_w, cam_role,
                      ego_speed_kmh=0.0):  
    active = [t for t in trackers if t.hit_streak >= CFG['min_hits']]
    n_vehicles = len(active)
    speeds_kmh = [t.speed_kmh for t in active]
    avg_speed  = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
    congested  = (n_vehicles >= CFG['congestion_count'] and
                  avg_speed  <= CFG['congestion_speed'])

    speed_var  = float(np.var(speeds_kmh)) if len(speeds_kmh) >= 2 else 0.0
    if   speed_var >= CFG['speed_variance_high']: var_level = 'DANGEROUS'
    elif speed_var >= CFG['speed_variance_med']:  var_level = 'MIXED'
    else: var_level = 'UNIFORM'
    tailgating   = False
    closest_dist = 0.0
    if active:
        bh_vals=[(t.get_state()[0][3]-t.get_state()[0][1])/max(frame_h,1) for t in active]
        closest_dist = float(max(bh_vals))
        tailgating   = closest_dist >= CFG['tailgate_box_h_ratio']
    braking_detected = False
    max_closing_rate = 0.0
    for t in active:
        if len(t.history_centers) >= 2:
            pts = np.array(t.history_centers)
            dh  = float(np.mean(np.diff(pts[:,2])))
            if dh > max_closing_rate: max_closing_rate = dh
            if dh >= CFG['braking_dh_thresh']: braking_detected = True
    proximity_score = min(closest_dist / max(CFG['tailgate_box_h_ratio'], 1e-3), 1.0) * 35
    closing_score   = min(max_closing_rate / 10.0,                                1.0) * 35
    variance_score  = min(speed_var / CFG['speed_variance_high'],                 1.0) * 30
    risk_score      = proximity_score + closing_score + variance_score
    if   risk_score >= CFG['risk_score_high']: risk_level = 'HIGH'
    elif risk_score >= CFG['risk_score_med']:  risk_level = 'MEDIUM'
    else:                                       risk_level = 'LOW'

    return {
        'cam_role'        : cam_role,
        'n_vehicles'      : n_vehicles,
        'avg_speed_kmh'   : avg_speed,
        'congested'       : congested,
        'speed_var'       : speed_var,
        'var_level'       : var_level,
        'tailgating'      : tailgating,
        'closest_dist'    : closest_dist,
        'braking'         : braking_detected,
        'closing_rate_px' : max_closing_rate,
        'risk_score'      : risk_score,
        'risk_level'      : risk_level,
        'ego_speed_kmh'   : ego_speed_kmh,   
    }

DASH_W, DASH_H = 720, 900

def draw_dashboard(analytics_all, metrics_all=None):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    dash[:] = (18, 18, 18)

    def txt(s, x, y, scale=0.55, color=(220,220,220), thick=1):
        cv2.putText(dash, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    def hbar(x, y, w, val, maxval, color):
        filled = int(w * min(val / max(maxval, 1e-3), 1.0))
        cv2.rectangle(dash, (x,y),         (x+w,      y+12), (45,45,45), -1)
        cv2.rectangle(dash, (x,y),         (x+filled, y+12), color,      -1)
        cv2.rectangle(dash, (x,y),         (x+w,      y+12), (90,90,90),  1)

    cv2.rectangle(dash, (0,0), (DASH_W,46), (28,28,28), -1)
    txt('TRAFFIC ANALYTICS DASHBOARD', 12, 32, 0.75, (0,210,255), 2)
    cv2.line(dash, (0,46), (DASH_W,46), (60,60,60), 1)
    cv2.line(dash, (DASH_W//2,46), (DASH_W//2,DASH_H-1), (55,55,55), 1)

    col_x = [16, DASH_W//2+16]

    for i, ana in enumerate(analytics_all):
        cx = col_x[i]
        if ana is None:
            txt('No signal', cx, 100, 0.5, (100,100,100))
            continue

        role = ana['cam_role']
        y = 72
        cv2.rectangle(dash, (cx-4,y-20), (cx+330,y+6), (35,35,35), -1)
        txt(f'[ {role} ]', cx, y, 0.62, (0,200,120), 2)
        y += 32
        es     = ana.get('ego_speed_kmh', 0.0)
        es_col = (0,200,255) if es < 60 else (50,50,255)
        cv2.rectangle(dash, (cx-4,y-16), (cx+330,y+28), (30,30,40), -1)
        cv2.rectangle(dash, (cx-4,y-16), (cx+330,y+28), (0,150,200), 1)
        txt('EGO VEHICLE (IMU)', cx, y, 0.48, (120,200,255))
        y += 20
        txt(f'{es:.1f} km/h', cx, y, 0.65, es_col, 2)
        hbar(cx+120, y-14, 200, es, 120.0, es_col)
        y += 32
        nv   = ana['n_vehicles']
        av   = ana['avg_speed_kmh']
        cong = ana['congested']
        cc   = (50,50,255) if cong else (80,200,80)
        txt(f'Vehicles: {nv}  |  Avg: {av:.0f} km/h', cx, y, 0.50, (200,200,200))
        y += 22
        txt('CONGESTED' if cong else 'FLOWING', cx, y, 0.60, cc, 2)
        hbar(cx, y+6, 320, nv, 10, cc)
        y += 42
        sv  = ana['speed_var']
        vl  = ana['var_level']
        vc  = (50,50,255) if vl=='DANGEROUS' else (0,170,255) if vl=='MIXED' else (80,255,80)
        txt(f'Speed Variance: {sv:.0f} km2/h2', cx, y, 0.50, (200,200,200))
        y += 22
        txt(vl, cx, y, 0.60, vc, 2)
        hbar(cx, y+6, 320, sv, CFG['speed_variance_high'], vc)
        y += 42
        cd  = ana['closest_dist']
        tg  = ana['tailgating']
        dc  = (50,50,255) if tg else (0,200,255) if cd > 0.2 else (80,255,80)
        txt(f'Closest: {cd*100:.0f}% frame height', cx, y, 0.50, (200,200,200))
        y += 22
        txt('!! TAILGATING !!' if tg else 'Safe Distance', cx, y, 0.60, dc, 2)
        hbar(cx, y+6, 320, cd, CFG['tailgate_box_h_ratio']*1.5, dc)
        y += 42
        cr  = ana['closing_rate_px']
        brk = ana['braking']
        bc  = (50,50,255) if brk else (0,200,255) if cr > 1.5 else (80,255,80)
        txt(f'Closing Rate: {cr:.1f} px/frame', cx, y, 0.50, (200,200,200))
        y += 22
        txt('!! BRAKING AHEAD !!' if brk else 'Normal', cx, y, 0.60, bc, 2)
        hbar(cx, y+6, 320, cr, 10, bc)
        y += 42
        rs  = ana['risk_score']
        rl  = ana['risk_level']
        rc  = (50,50,255) if rl=='HIGH' else (0,170,255) if rl=='MEDIUM' else (80,255,80)
        txt(f'Risk Score: {rs:.0f} / 100', cx, y, 0.50, (200,200,200))
        y += 22
        txt(f'[ {rl} ]', cx, y, 0.65, rc, 2)
        hbar(cx, y+8, 320, rs, 100, rc)
    cv2.line(dash, (0,DASH_H-26), (DASH_W,DASH_H-26), (40,40,40), 1)
    if metrics_all:
        METRICS_Y_START = 730
        cv2.line(dash, (0, METRICS_Y_START - 8), (DASH_W, METRICS_Y_START - 8), (60,60,60), 1)
        cv2.rectangle(dash, (0, METRICS_Y_START - 24), (DASH_W, METRICS_Y_START - 4), (28,28,28), -1)
        txt('EDGE AI PERFORMANCE METRICS', 12, METRICS_Y_START - 8, 0.60, (0,210,255), 2)
        metric_col_x = [16, DASH_W // 2 + 16]
        metric_labels = [
            ('Inf Latency (avg/p95)',  'avg_inference_ms',  'p95_inference_ms',  'ms'),
            ('Frame Latency (avg/p95)','avg_frame_lat_ms',  'p95_frame_lat_ms',  'ms'),
            ('Throughput',             'throughput_fps',     None,                'FPS'),
            ('Model Footprint',        'model_footprint_mb', None,               'MB'),
            ('Peak RSS Memory',        'peak_rss_mb',        None,               'MB'),
            ('Runtime Heap (cur/peak)','cur_heap_mb',        'peak_heap_mb',     'MB'),
            ('CPU Usage (avg/peak)',   'avg_cpu_pct',        'peak_cpu_pct',     '%'),
        ]
        for i, m in enumerate(metrics_all):
            if m is None:
                continue
            cx = metric_col_x[i]
            cam_label = NICLA_CAMERAS[i].get('role', NICLA_CAMERAS[i]['name'])
            y = METRICS_Y_START + 14
            cv2.rectangle(dash, (cx-4, y-14), (cx+330, y+4), (35,35,35), -1)
            txt(f'[ {cam_label} ]', cx, y, 0.48, (0,200,120), 1)
            y += 16
            for label, key1, key2, unit in metric_labels:
                v1 = m.get(key1, 0.0)
                txt_label = f'{label}:'
                if key2:
                    v2 = m.get(key2, 0.0)
                    val_str = f'{v1:.1f} / {v2:.1f} {unit}'
                else:
                    val_str = f'{v1:.1f} {unit}'
                txt(txt_label, cx, y, 0.38, (130,130,130))
                txt(val_str,   cx + 158, y, 0.40, (200,255,200))
                y += 14

    txt(f'Updated: {time.strftime("%H:%M:%S")}', 12, DASH_H-8, 0.42, (70,70,70))
    txt('CAM 0 = FRONT   CAM 1 = REAR', DASH_W//2-120, DASH_H-8, 0.42, (70,70,70))
    return dash

quit_flag        = threading.Event()
frame_queues     = {cam['id']: queue.Queue(maxsize=1) for cam in NICLA_CAMERAS}
analytics_queues = {cam['id']: queue.Queue(maxsize=1) for cam in NICLA_CAMERAS}
metrics_queues   = {cam['id']: queue.Queue(maxsize=1) for cam in NICLA_CAMERAS}

def camera_worker(cam_cfg):
    cam_id     = cam_cfg["id"]
    cam_name   = cam_cfg["name"]
    cam_role   = cam_cfg.get("role", cam_name)
    stream_url = f"http://{cam_cfg['ip']}:{cam_cfg['port']}"
    fq         = frame_queues[cam_id]
    aq         = analytics_queues[cam_id]
    print(f"[{cam_name}] Starting... URL: {stream_url}")
    stream = NiclaStream(stream_url)
    model  = YOLO(CFG['yolo_model'])
    metrics = EdgeAIMetrics(cam_name)
    metrics.set_model_footprint(CFG['yolo_model'])
    print(f"[{cam_name}] Model footprint: {metrics.model_footprint_mb:.2f} MB")
    tracker   = SORTTracker()
    prev_gray = None
    ego_px    = 0.0
    last_dets = np.empty((0, 4))
    last_cls  = []
    frame_idx = 0
    t_start   = time.time()
    fps       = CFG['assumed_fps']
    width = height = None
    print(f"[{cam_name}] Processing thread running")
    while not quit_flag.is_set():
        _frame_start = time.perf_counter()      
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame_idx == 0:
            height, width = frame.shape[:2]
            print(f"[{cam_name}] Resolution: {width}x{height}")
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and CFG['use_egomotion']:
            ego_px = estimate_ego_motion(prev_gray, curr_gray)
        ego_speed_ms  = stream.get_ego_speed_ms()     
        ego_speed_kmh = ego_speed_ms * 3.6             
        if frame_idx % CFG['yolo_every_n'] == 0:
            _inf_start = time.perf_counter()        # ← inference latency start
            results   = model(frame, imgsz=CFG['yolo_imgsz'], conf=CFG['conf_thresh'],
                              iou=CFG['iou_thresh'], classes=CFG['target_classes'],
                              verbose=False)[0]
            metrics.record_inference((time.perf_counter() - _inf_start) * 1000)  # ← record

            last_dets = np.empty((0, 4)); last_cls = []
            if results.boxes is not None and len(results.boxes):
                dets, cls_list = [], []
                for b in results.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    x1=max(0,x1); y1=max(0,y1)
                    x2=min(width-1,x2); y2=min(height-1,y2)
                    dets.append([x1,y1,x2,y2]); cls_list.append(int(b.cls[0]))
                last_dets = np.array(dets, dtype=float); last_cls = cls_list
        active = tracker.update(last_dets, last_cls)
        for trk in active:
            if trk.hit_streak >= CFG['min_hits']:
                trk.update_speed(ego_px, fps)
        elapsed  = time.time() - t_start
        fps_disp = frame_idx / max(elapsed, 0.001)
        annotated = draw_overlay(frame.copy(), active, ego_px, fps_disp,
                                 frame_idx, cam_name,
                                 ego_speed_kmh=ego_speed_kmh) 
        if height and width:
            ana = compute_analytics(active, ego_px, fps, height, width, cam_role,
                                    ego_speed_kmh=ego_speed_kmh)
            try: aq.get_nowait()
            except queue.Empty: pass
            aq.put_nowait(ana)
            mq = metrics_queues[cam_id]
            try: mq.get_nowait()
            except queue.Empty: pass
            mq.put_nowait(metrics.snapshot())
        disp_h  = int(height * CFG['display_width'] / width)
        display = cv2.resize(annotated, (CFG['display_width'], disp_h))
        try: fq.get_nowait()
        except queue.Empty: pass
        fq.put_nowait(display)
        prev_gray  = curr_gray.copy()
        frame_idx += 1
        metrics.record_frame_latency((time.perf_counter() - _frame_start) * 1000)
        if frame_idx % 300 == 0:
            s = metrics.snapshot()
            print(f"[{cam_name}][METRICS] "
                  f"inf={s['avg_inference_ms']:.1f}ms "
                  f"lat={s['avg_frame_lat_ms']:.1f}ms "
                  f"fps={s['throughput_fps']:.1f} "
                  f"rss={s['peak_rss_mb']:.0f}MB "
                  f"cpu={s['avg_cpu_pct']:.1f}%")
    stream.stop()
    total_t = time.time() - t_start
    print(f"\n[{cam_name}] {frame_idx} frames in {total_t:.1f}s "
          f"(avg {frame_idx/max(total_t,0.001):.1f} fps)")
    metrics.print_report()

def main():
    print(f"Dual Nicla Vision Speed Tracker — {len(NICLA_CAMERAS)} cameras")
    window_names = {}
    for cam_cfg in NICLA_CAMERAS:
        wname = f"Nicla [{cam_cfg['role']}] {cam_cfg['name']} — (Q to quit)"
        window_names[cam_cfg['id']] = wname
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, CFG['display_width'], CFG['display_width']*9//16)
        print(f"{wname}")
    DASH_WIN = 'Traffic Analytics Dashboard (Q to quit)'
    cv2.namedWindow(DASH_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DASH_WIN, DASH_W, DASH_H)
    print(f"Dashboard window ready")
    threads = []
    for cam_cfg in NICLA_CAMERAS:
        t = threading.Thread(target=camera_worker, args=(cam_cfg,), daemon=True)
        t.start(); threads.append(t)
        print(f"Thread started: {cam_cfg['name']} @ {cam_cfg['ip']}:{cam_cfg['port']}")
    print("Display loop running — Press Q or ESC to quit")
    latest_ana = [None, None]
    latest_met = [None, None]
    while not quit_flag.is_set():
        for cam_cfg in NICLA_CAMERAS:
            cid = cam_cfg['id']
            try:
                frame = frame_queues[cid].get_nowait()
                cv2.imshow(window_names[cid], frame)
            except queue.Empty:
                pass
        for cam_cfg in NICLA_CAMERAS:
            cid = cam_cfg['id']
            try: latest_ana[cid] = analytics_queues[cid].get_nowait()
            except queue.Empty: pass
            try: latest_met[cid] = metrics_queues[cid].get_nowait()
            except queue.Empty: pass
        cv2.imshow(DASH_WIN, draw_dashboard(latest_ana, metrics_all=latest_met))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Quit by user.")
            quit_flag.set()
            break
    for t in threads:
        t.join(timeout=3)
    cv2.destroyAllWindows()
    print("\nAll sessions ended.")

if __name__ == '__main__':
    main()