import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import List, Dict, Tuple
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet.models import EmoNet
import cv2
import sys
import datetime
import os
from PIL import Image, ImageDraw, ImageFont
import decord
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_video(video_path: Path, max_frames: int = -1, target_fps: float = 15.0) -> Tuple[np.ndarray, List[float], float]:
    """优化后的视频加载函数，使用Decord库实现高效帧采样"""
    if not video_path.exists():
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    try:
        vr = decord.VideoReader(str(video_path))
    except Exception as e:
        raise RuntimeError(f"无法打开视频文件: {video_path} ({str(e)})")

    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / original_fps

    print(f"视频解析: 总帧数={total_frames}, 原始FPS={original_fps:.2f}, 时长={duration:.2f}秒")

    # 计算采样时间点
    sample_interval = 1.0 / target_fps
    sample_times = np.arange(0, duration, sample_interval)
    if max_frames > 0 and len(sample_times) > max_frames:
        sample_times = sample_times[:max_frames]

    # 转换为帧索引并去重
    frame_indices = np.clip((sample_times * original_fps).astype(int), 0, total_frames-1)
    frame_indices = np.unique(frame_indices)

    # 批量读取帧数据
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        raise RuntimeError(f"帧读取失败: {str(e)}")

    timestamps = frame_indices / original_fps
    actual_fps = 1/np.mean(np.diff(timestamps)) if len(timestamps)>1 else target_fps

    print(f"采样完成: 有效帧数={len(frames)}, 实际FPS={actual_fps:.2f}")
    return frames, timestamps.tolist(), actual_fps

def load_emonet(n_expression: int, device: str) -> nn.Module:
    """优化后的模型加载函数，支持半精度"""
    state_dict_path = Path(__file__).parent.joinpath("pretrained", f"emonet_{n_expression}.pth")
    print(f"加载EmoNet模型: {state_dict_path}")

    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model = EmoNet(n_expression=n_expression).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model.half().eval()  # 转换为半精度

def batch_detect_faces(
    detector: SFDDetector,
    frames: List[np.ndarray],
    batch_size: int = 32,
    detect_size: Tuple[int, int] = (256, 256)
) -> List[List[np.ndarray]]:
    """批量人脸检测函数，支持多线程和缩略图处理"""
    def process_frame(frame: np.ndarray) -> List[np.ndarray]:
        scaled = cv2.resize(frame, detect_size)
        return detector.detect_from_image(scaled[:, :, ::-1])  # RGB转BGR

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(process_frame, frames),
            total=len(frames),
            desc="人脸检测",
            unit="frame"
        ))
    return results

def batch_emotion_analysis(
    model: nn.Module,
    frames: List[np.ndarray],
    bboxes: List[List[int]],
    image_size: int = 256,
    batch_size: int = 64
) -> Tuple[List[str], List[float], List[float]]:
    """批量情绪分析函数"""
    device = next(model.parameters()).device
    face_tensors = []
    valid_indices = []

    # 准备人脸数据
    for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            face = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if face.size == 0:
                continue
            face = cv2.resize(face, (image_size, image_size))
            face_tensor = torch.from_numpy(face).permute(2,0,1).half()/255.0
            face_tensors.append(face_tensor)
            valid_indices.append(idx)
        except:
            continue

    # 批量推理
    emotions, valences, arousals = [], [], []
    for i in range(0, len(face_tensors), batch_size):
        batch = torch.stack(face_tensors[i:i+batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
        
        # 解析结果
        probs = torch.softmax(outputs["expression"], dim=1)
        batch_emotions = probs.argmax(dim=1).cpu().numpy()
        batch_valences = outputs["valence"].cpu().numpy().flatten()
        batch_arousals = outputs["arousal"].cpu().numpy().flatten()

        emotions.extend(batch_emotions)
        valences.extend(batch_valences)
        arousals.extend(batch_arousals)

    return emotions, valences, arousals, valid_indices

def format_time(seconds: float, start_time: datetime.datetime = None) -> str:
    """优化后的时间格式化函数"""
    if start_time:
        return (start_time + datetime.timedelta(seconds=seconds)).strftime("%H:%M:%S.%f")[:-3]
    return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}"

def main():
    parser = argparse.ArgumentParser("高效视频情绪分析系统", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nclasses", type=int, choices=[5,8], default=8, help="情绪分类数")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_path", type=str, default="emotion_analysis.xlsx", help="输出文件路径")
    parser.add_argument("--max_frames", type=int, default=-1, help="最大处理帧数")
    parser.add_argument("--start_time", type=str, help="起始时间(格式: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output_frames", type=str, help="标注帧输出目录")
    parser.add_argument("--target_fps", type=float, default=15.0, help="目标采样率")
    parser.add_argument("--batch_size", type=int, default=64, help="处理批大小")
    args = parser.parse_args()

    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emotion_map = ["中性", "快乐", "悲伤", "惊讶", "恐惧", "厌恶", "愤怒", "轻蔑"]
    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S") if args.start_time else None
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 加载视频数据
        frames, timestamps, actual_fps = load_video(
            Path(args.video_path),
            max_frames=args.max_frames,
            target_fps=args.target_fps
        )

        # 初始化模型
        detector = SFDDetector(device)
        emonet = load_emonet(args.nclasses, device)

        # 批量人脸检测
        detect_results = batch_detect_faces(detector, frames)

        # 准备分析结果
        data = []
        bboxes = []
        for idx, (timestamp, detections) in enumerate(zip(timestamps, detect_results)):
            if len(detections) > 0:
                bbox = detections[0][:4] * (frames[idx].shape[1]/256, frames[idx].shape[0]/256)*2
                bboxes.append(bbox.astype(int).tolist())
            else:
                bboxes.append([])
            data.append({
                "时间戳": format_time(timestamp, start_time),
                "绝对时间": (start_time + datetime.timedelta(seconds=timestamp)).isoformat() if start_time else None,
                "相对时间(s)": timestamp,
                "帧序号": idx,
                "情绪": None,
                "效价": None,
                "唤醒度": None,
                "检测状态": len(detections) > 0
            })

        # 批量情绪分析
        valid_frames = [frames[i] for i, b in enumerate(bboxes) if len(b) > 0]
        valid_bboxes = [b for b in bboxes if len(b) > 0]
        emotions, valences, arousals, valid_idx = batch_emotion_analysis(emonet, valid_frames, valid_bboxes)

        # 更新有效结果
        for idx, e, v, a in zip(valid_idx, emotions, valences, arousals):
            data[idx].update({
                "情绪": emotion_map[e],
                "效价": float(v),
                "唤醒度": float(a)
            })

        # 保存结果
        df = pd.DataFrame(data)
        with pd.ExcelWriter(args.output_path, engine="openpyxl") as writer:
            pd.DataFrame([{
                "分析时间": analysis_time,
                "视频路径": args.video_path,
                "采样帧率": actual_fps,
                "处理帧数": len(frames),
                "设备类型": device
            }]).to_excel(writer, sheet_name="分析摘要", index=False)
            df.to_excel(writer, sheet_name="情绪数据", index=False)

        print(f"分析完成，结果已保存至: {args.output_path}")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()