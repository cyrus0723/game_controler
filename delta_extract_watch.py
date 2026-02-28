import time
from pathlib import Path

import cv2
import numpy as np
from mss import mss
from plyer import notification

# =========================
# 固定配置（按你的情况）
# =========================
SCREEN_W, SCREEN_H = 1920, 1080

# 结算标题所在区域（只截这里，速度快、误报低）
ROI = {"left": 150, "top": 150, "width": 500, "height": 160}

# 模板文件
TEMPL_DIR = Path("templates")
T_WIN = TEMPL_DIR / "success.png"
T_LOSE = TEMPL_DIR / "fail.png"

# 匹配阈值：0.78~0.90 之间调；先用 0.82
THRESHOLD = 0.82

# 同一局结算避免反复提醒
COOLDOWN_SECONDS = 25

# 扫描频率
SCAN_INTERVAL = 0.20
# =========================


def grab_roi(sct: mss, roi: dict) -> np.ndarray:
    """抓取 ROI，返回 BGR 图像"""
    shot = sct.grab(roi)
    frame = np.array(shot)[:, :, :3]  # BGRA->BGR
    return frame


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """轻量预处理：灰度 + 轻微去噪"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"读不到模板：{path}")
    return img


def match_score(screen_gray: np.ndarray, templ_gray: np.ndarray) -> float:
    res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def capture_template(name: str, out_path: Path):
    """在结算画面时运行此函数，采集 ROI 保存为模板"""
    TEMPL_DIR.mkdir(parents=True, exist_ok=True)
    with mss() as sct:
        frame = grab_roi(sct, ROI)
    cv2.imwrite(str(out_path), frame)
    print(f"[OK] 已保存 {name} 模板到：{out_path}\n"
          f"提示：请确保此时画面正处于{name}结算页，且ROI内包含“撤离成功/撤离失败”等固定文字。")


def notify(result: str, score: float):
    notification.notify(
        title="三角洲行动：检测到结算",
        message=f"{result}（匹配度 {score:.2f}）\n打完这把就停手？",
        timeout=5
    )


def watch():
    templ_win = load_gray(T_WIN)
    templ_lose = load_gray(T_LOSE)

    last_trigger = 0.0

    print("开始监测结算标题区域（ROI）...")
    print(f"ROI = {ROI}")
    print("按 Ctrl+C 退出。\n")

    with mss() as sct:
        while True:
            now = time.time()
            if now - last_trigger < COOLDOWN_SECONDS:
                time.sleep(SCAN_INTERVAL)
                continue

            frame = grab_roi(sct, ROI)
            gray = preprocess(frame)

            s_win = match_score(gray, templ_win)
            s_lose = match_score(gray, templ_lose)

            best = max(s_win, s_lose)
            if best >= THRESHOLD:
                result = "撤离成功" if s_win >= s_lose else "撤离失败"
                notify(result, best)
                print(f"[触发] {result} score={best:.2f}")
                last_trigger = now

            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["success", "fail"], help="采集模板：在对应结算页运行")
    args = parser.parse_args()

    if args.capture == "success":
        capture_template("撤离成功", T_WIN)
    elif args.capture == "fail":
        capture_template("撤离失败", T_LOSE)
    else:
        if not T_WIN.exists() or not T_LOSE.exists():
            print("缺少模板！请先采集：")
            print("  python delta_extract_watch.py --capture success   （在撤离成功结算页执行）")
            print("  python delta_extract_watch.py --capture fail      （在撤离失败结算页执行）")
            raise SystemExit(1)
        watch()