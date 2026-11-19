import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- AYARLAR ---
VIDEO_PATH = "video_2.mp4"      # Videonun dosya adı (veya yolu)
TRACK_LENGTH_CM = 80.0           # Eğik düzlem için 93.0, çarpışma için 80.0
EXPERIMENT_TYPE = "collision"    # "incline" (eğik düzlem) veya "collision" (çarpışma)
INCLINE_HEIGHT_CM = 7.5          # Sadece eğik düzlem deneyi için

# HSV Renk Aralıkları
LOWER_BLUE = np.array([95, 110, 70])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_RED1 = np.array([0, 110, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 110, 80])
UPPER_RED2 = np.array([179, 255, 255])

cal_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(cal_points) < 2:
        cal_points.append((x, y))
        cv2.circle(param, (x, y), 6, (0, 255, 0), -1)
        if len(cal_points) == 2:
            cv2.line(param, cal_points[0], cal_points[1], (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon - Rayin Iki Ucunu Secin", param)

def get_centroid(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 150:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def project_to_track(pt, p1, u_vec, scale_cm_per_px):
    vec = np.array([pt[0] - p1[0], pt[1] - p1[1]], dtype=float)
    s_px = np.dot(vec, u_vec)
    return s_px * scale_cm_per_px

def smooth_series(arr, window=5):
    """Piksel titremesini azaltarak temiz hız türevi almayı sağlar."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

# Çıktıların kodun bulunduğu klasöre kaydedilmesini garantile
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError(f"Video acilamadi: {VIDEO_PATH}")

# 1. Kalibrasyon Adımı
clone = first_frame.copy()
cv2.imshow("Kalibrasyon - Rayin Iki Ucunu Secin", clone)
cv2.setMouseCallback("Kalibrasyon - Rayin Iki Ucunu Secin", mouse_callback, clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(cal_points) != 2:
    raise ValueError("Rayin iki ucunu secmelisiniz.")

p1, p2 = np.array(cal_points[0], dtype=float), np.array(cal_points[1], dtype=float)
track_vec = p2 - p1
track_px = np.linalg.norm(track_vec)
u_vec = track_vec / track_px
scale = TRACK_LENGTH_CM / track_px

roi_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
cv2.line(roi_mask, tuple(cal_points[0]), tuple(cal_points[1]), 255, thickness=90)

# 2. Kare Kare Takip
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
data = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = frame_idx / fps
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
        cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    )
    mask_red = cv2.bitwise_and(mask_red, roi_mask)
    c_red = get_centroid(mask_red)

    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_blue = cv2.bitwise_and(mask_blue, roi_mask)
    c_blue = get_centroid(mask_blue)

    s_red = project_to_track(c_red, p1, u_vec, scale) if c_red else np.nan
    s_blue = project_to_track(c_blue, p1, u_vec, scale) if c_blue else np.nan
    data.append((t, s_blue, s_red))

    # Canlı HUD Görselleştirme
    cv2.line(frame, tuple(cal_points[0]), tuple(cal_points[1]), (0, 255, 0), 2)
    if c_red:
        cv2.circle(frame, c_red, 6, (0, 0, 255), -1)
    if c_blue:
        cv2.circle(frame, c_blue, 6, (255, 0, 0), -1)
    cv2.putText(frame, f"Sure: {t:.2f} s", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Deney Takibi", frame)
    if cv2.waitKey(15) & 0xFF == 27:
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
data = np.array(data)

# 3. Hesaplamalar, CSV Kaydı ve Grafik Çizimi
csv_path = os.path.join(BASE_DIR, "deney_verileri.csv")
png_path = os.path.join(BASE_DIR, "deney_grafikleri.png")

if EXPERIMENT_TYPE == "incline":
    valid = ~np.isnan(data[:, 2])
    t_arr = data[valid, 0]
    s_red_cm = data[valid, 2]
    s_red_m = s_red_cm / 100.0

    # Hız dizisi (m/s)
    v_red = np.gradient(smooth_series(s_red_m, 7), t_arr)

    s_min, s_max = np.min(s_red_cm), np.max(s_red_cm)
    span = s_max - s_min
    active = (s_red_cm > s_min + 0.10 * span) & (s_red_cm < s_max - 0.15 * span)

    coeffs = np.polyfit(t_arr[active], s_red_m[active], 2)
    a_exp = abs(2 * coeffs[0])
    a_theo = 9.81 * (INCLINE_HEIGHT_CM / TRACK_LENGTH_CM)
    err_pct = abs(a_theo - a_exp) / a_theo * 100

    print("\n--- EGIK DUZLEM SONUCLARI ---")
    print(f"Deneysel Ivme : {a_exp:.3f} m/s^2")
    print(f"Teorik Ivme   : {a_theo:.3f} m/s^2")
    print(f"Bagil Hata    : %{err_pct:.2f}")

    # CSV Kaydet (Türkçe Excel uyumlu noktalı virgül ayırıcı)
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Zaman (s)", "Mavi_Konum (cm)", "Kirmizi_Konum (cm)", "Mavi_Hiz (m/s)", "Kirmizi_Hiz (m/s)"])
        for t_val, s_val, v_val in zip(t_arr, s_red_cm, v_red):
            writer.writerow([f"{t_val:.4f}", "-", f"{s_val:.3f}", "-", f"{v_val:.4f}"])

    # Grafikler
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(t_arr, s_red_cm, color="crimson", linewidth=2, label="Kırmızı Araba Konumu")
    ax1.set_title("Konum - Zaman Grafiği s(t)")
    ax1.set_xlabel("Zaman (s)")
    ax1.set_ylabel("Konum (cm)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(t_arr, v_red, color="darkred", linewidth=2, label=f"Hız (a_deney = {a_exp:.2f} m/s²)")
    ax2.set_title("Hız - Zaman Grafiği v(t)")
    ax2.set_xlabel("Zaman (s)")
    ax2.set_ylabel("Hız (m/s)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

elif EXPERIMENT_TYPE == "collision":
    valid = (~np.isnan(data[:, 1])) & (~np.isnan(data[:, 2]))
    t_arr = data[valid, 0]
    s_blue_cm = data[valid, 1]
    s_red_cm = data[valid, 2]
    s_blue_m = s_blue_cm / 100.0
    s_red_m = s_red_cm / 100.0

    # Anlık hız dizileri (m/s)
    v_blue = np.gradient(smooth_series(s_blue_m, 7), t_arr)
    v_red = np.gradient(smooth_series(s_red_m, 7), t_arr)

    dist = np.abs(s_red_m - s_blue_m)
    col_idx = np.argmin(dist)
    t_col = t_arr[col_idx]

    pre = (t_arr > t_col - 1.0) & (t_arr < t_col - 0.25)
    post = (t_arr > t_col + 0.25) & (t_arr < t_col + 1.0)

    v_blue_pre = np.polyfit(t_arr[pre], s_blue_m[pre], 1)[0]
    v_red_pre = np.polyfit(t_arr[pre], s_red_m[pre], 1)[0]
    v_blue_post = np.polyfit(t_arr[post], s_blue_m[post], 1)[0]
    v_red_post = np.polyfit(t_arr[post], s_red_m[post], 1)[0]

    e = abs(v_red_post - v_blue_post) / abs(v_blue_pre - v_red_pre)
    print("\n--- CARPISMA DENEYI SONUCLARI ---")
    print(f"Carpisma Ani            : {t_col:.2f} sn")
    print(f"Once  -> Mavi: {v_blue_pre:.3f} m/s | Kirmizi: {v_red_pre:.3f} m/s")
    print(f"Sonra -> Mavi: {v_blue_post:.3f} m/s | Kirmizi: {v_red_post:.3f} m/s")
    print(f"Esneklik Katsayisi (e)  : {e:.2f}")

    # CSV Kaydet (Türkçe Excel uyumlu)
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Zaman (s)", "Mavi_Konum (cm)", "Kirmizi_Konum (cm)", "Mavi_Hiz (m/s)", "Kirmizi_Hiz (m/s)"])
        for t_v, sb, sr, vb, vr in zip(t_arr, s_blue_cm, s_red_cm, v_blue, v_red):
            writer.writerow([f"{t_v:.4f}", f"{sb:.3f}", f"{sr:.3f}", f"{vb:.4f}", f"{vr:.4f}"])

    # Yan Yana İki Grafik
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Konum - Zaman Grafiği
    ax1.plot(t_arr, s_blue_cm, color="royalblue", linewidth=2, label="Mavi Araba")
    ax1.plot(t_arr, s_red_cm, color="crimson", linewidth=2, label="Kırmızı Araba")
    ax1.axvline(t_col, color="gray", linestyle="--", label=f"Çarpışma ({t_col:.2f} s)")
    ax1.set_title("Konum - Zaman Grafiği s(t)")
    ax1.set_xlabel("Zaman (s)")
    ax1.set_ylabel("Konum (cm)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Hız - Zaman Grafiği
    ax2.plot(t_arr, v_blue, color="royalblue", linewidth=2, label="Mavi Hız")
    ax2.plot(t_arr, v_red, color="crimson", linewidth=2, label="Kırmızı Hız")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvline(t_col, color="gray", linestyle="--", label="Çarpışma Anı")
    ax2.set_title("Hız - Zaman Grafiği v(t)")
    ax2.set_xlabel("Zaman (s)")
    ax2.set_ylabel("Hız (m/s)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

plt.tight_layout()
plt.savefig(png_path, dpi=200)
print(f"\nDosyalar basariyla kaydedildi:\n- {csv_path}\n- {png_path}")
plt.show()