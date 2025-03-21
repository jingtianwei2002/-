import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from typing import List, Dict
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
import numpy as np


def load_video(video_path: Path, max_frames: int = -1, target_fps: float = 15.0) -> tuple:
    """
    加载视频并返回帧和时间戳，使用基于时间戳的采样实现目标帧率
    """
    if not video_path.exists():
        raise FileNotFoundError(f"未找到视频文件: {video_path}")
        
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
        
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    duration = total_frames / original_fps
    
    if total_frames == 0 or original_fps == 0:
        raise RuntimeError(f"无效的视频文件: 总帧数={total_frames}, FPS={original_fps}")
    
    print(f"视频共有 {total_frames} 帧, 原始FPS: {original_fps:.2f}, 时长: {duration:.2f}秒")
    
    # 计算理论上需要的采样点
    sample_interval = 1.0 / target_fps  # 采样间隔（秒）
    sample_times = np.arange(0, duration, sample_interval)
    if max_frames > 0:
        sample_times = sample_times[:max_frames]
    
    list_frames_rgb = []
    timestamps = []
    last_frame = None
    last_frame_time = -1

    for target_time in sample_times:
        # 定位到目标时间点的帧
        video_capture.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ret, frame = video_capture.read()
        
        if not ret:
            break
            
        # 获取实际的时间戳
        actual_time = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 转换为RGB并保存
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        list_frames_rgb.append(image_rgb)
        timestamps.append(actual_time)
        
        if len(list_frames_rgb) % 100 == 0:
            print(f"已加载 {len(list_frames_rgb)} 帧...")
    
    video_capture.release()
    
    if not list_frames_rgb:
        raise RuntimeError("未能从视频中加载任何帧")
        
    actual_fps = len(list_frames_rgb) / timestamps[-1]
    print(f"共加载了 {len(list_frames_rgb)} 帧，实际采样帧率: {actual_fps:.2f} fps")
    return list_frames_rgb, timestamps, actual_fps


def load_emonet(n_expression: int, device: str):
    """
    加载情绪识别模型
    """
    state_dict_path = Path(__file__).parent.joinpath(
        "pretrained", f"emonet_{n_expression}.pth"
    )

    print(f"正在加载emonet模型: {state_dict_path}")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    return net


def run_emonet(
    emonet: torch.nn.Module, frame_rgb: np.ndarray, image_size: int, device: str
) -> Dict[str, torch.Tensor]:
    """
    对单帧运行情绪识别
    """
    image_rgb = cv2.resize(frame_rgb, (image_size, image_size))
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0

    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))

    return output


def format_time(seconds: float, start_time: datetime.datetime = None) -> str:
    """
    将秒数转换为绝对时间格式
    """
    if start_time:
        time_point = start_time + datetime.timedelta(seconds=seconds)
        return time_point.strftime("%H:%M:%S.%f")[:-3]
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def draw_emotion_results(
    frame: np.ndarray,
    bbox: np.ndarray,
    emotion: str,
    valence: float,
    arousal: float,
    timestamp: str,
) -> np.ndarray:
    """
    在图片上绘制情绪分析结果，支持中文显示
    """
    img = frame.copy()
    # 绘制人脸框
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # 转换为PIL图像以支持中文
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 32)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 32)
        except:
            print("警告：未找到合适的中文字体，将使用默认字体")
            font = ImageFont.load_default()
    
    # 准备文本
    texts = [
        f"时间: {timestamp}",
        f"情绪: {emotion}",
        f"效价: {valence:.2f}",
        f"唤醒度: {arousal:.2f}"
    ]
    
    # 在左上角绘制文本
    margin_left = 10  # 左边距
    y_offset = 10    # 顶部边距
    green_color = (0, 255, 0)  # RGB格式的绿色
    
    for text in texts:
        # 获取文本大小
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制黑色背景
        draw.rectangle(
            [margin_left, y_offset, margin_left + text_width, y_offset + text_height],
            fill=(0, 0, 0)
        )
        
        # 绘制绿色文本
        draw.text((margin_left, y_offset), text, fill=green_color, font=font)
        y_offset += text_height + 5
    
    # 转换回OpenCV格式
    return np.array(img_pil)


def main():
    parser = argparse.ArgumentParser(description='视频情绪分析处理')
    parser.add_argument(
        '--nclasses',
        type=int,
        default=8,
        choices=[5, 8],
        help='情绪类别数量 (5 或 8)'
    )
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='输入视频文件路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='emotion_analysis.xlsx',
        help='输出Excel文件路径'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=-1,
        help='最大处理帧数 (-1 表示处理所有帧)'
    )
    parser.add_argument(
        '--start_time',
        type=str,
        help='视频起始时间点 (格式: YYYY-MM-DD HH:MM:SS)',
        default=None
    )
    parser.add_argument(
        '--output_frames',
        type=str,
        help='输出帧图片的目录路径',
        default=None
    )
    # 移除 frame_step 参数，增加 target_fps 参数
    parser.add_argument(
        '--target_fps',
        type=float,
        default=15.0,
        help='目标采样帧率 (默认: 15.0)'
    )
    
    args = parser.parse_args()

    try:
        # 检查是否安装了openpyxl
        import importlib
        if importlib.util.find_spec("openpyxl") is None:
            print("错误: 未安装openpyxl. 请使用以下命令安装:")
            print("pip install openpyxl")
            sys.exit(1)

        # 验证视频路径
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"错误: 未找到视频文件: {video_path}")
            sys.exit(1)

        # 实验参数设置
        n_expression = args.nclasses
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        image_size = 256
        emotion_classes = {
            0: "中性",
            1: "快乐",
            2: "悲伤",
            3: "惊讶",
            4: "恐惧",
            5: "厌恶",
            6: "愤怒",
            7: "轻蔑",
        }

        # 记录分析开始时间
        analysis_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 处理起始时间
        start_time = None
        if args.start_time:
            try:
                start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
                print(f"使用指定的起始时间: {start_time}")
            except ValueError:
                print("错误：起始时间格式不正确，应为 YYYY-MM-DD HH:MM:SS")
                sys.exit(1)

        print(f"加载emonet模型")
        emonet = load_emonet(n_expression, device)

        print(f"加载人脸检测器")
        sfd_detector = SFDDetector(device)

        print(f"加载视频并按 {args.target_fps} fps采样: {video_path}")
        list_frames_rgb, timestamps, actual_fps = load_video(
            video_path, 
            max_frames=args.max_frames,
            target_fps=args.target_fps
        )

        # 创建输出图片目录
        if args.output_frames:
            output_frames_dir = Path(args.output_frames)
            output_frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"将保存标注后的视频帧到: {output_frames_dir}")

        # 准备数据
        data = []

        for i, (frame, timestamp) in enumerate(zip(list_frames_rgb, timestamps)):
            with torch.no_grad():
                detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])

            if len(detected_faces) > 0:
                bbox = np.array(detected_faces[0]).astype(np.int32)
                face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                emotion_prediction = run_emonet(emonet, face_crop.copy(), image_size, device)
                
                emotion_probs = nn.functional.softmax(emotion_prediction["expression"], dim=1)
                predicted_emotion = emotion_classes[torch.argmax(emotion_probs).item()]
                
                valence = emotion_prediction["valence"].item()
                arousal = emotion_prediction["arousal"].item()
                
            
                data.append({
                    '时间点': format_time(timestamp, start_time),
                    '时间戳(秒)': timestamp,
                    '时间': (start_time + datetime.timedelta(seconds=timestamp)).strftime("%Y-%m-%d %H:%M:%S") if start_time else None,
                    '帧号': i,
                    '情绪': predicted_emotion,
                    '效价': valence,
                    '唤醒度': arousal,
                    '是否检测到人脸': True
                })
            else:
                data.append({
                    '时间点': format_time(timestamp, start_time),
                    '时间戳(秒)': timestamp,
                    '时间': (start_time + datetime.timedelta(seconds=timestamp)).strftime("%Y-%m-%d %H:%M:%S") if start_time else None,
                    '帧号': i,
                    '情绪': '未检测到人脸',
                    '效价': None,
                    '唤醒度': None,
                    '是否检测到人脸': False
                })

            if i % 100 == 0:
                print(f"已处理 {i}/{len(list_frames_rgb)} 帧")

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        # 添加分析信息
        analysis_info = pd.DataFrame([{
            '分析时间': analysis_start_time,
            '视频路径': str(video_path),
            '视频起始时间': start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "未指定",
            '视频总帧数': len(list_frames_rgb),
            '视频FPS': actual_fps,
            '处理步长': args.target_fps,
            '情绪类别数': n_expression
        }])

        output_path = Path(args.output_path)
        
        try:
            # 创建ExcelWriter对象
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 保存分析信息到第一个sheet
                analysis_info.to_excel(writer, sheet_name='分析信息', index=False)
                # 保存情绪数据到第二个sheet
                df.to_excel(writer, sheet_name='情绪数据', index=False)
            print(f"结果已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")
            # 尝试保存为CSV作为备选
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"结果已保存为CSV文件: {csv_path}")

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
