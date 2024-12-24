from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import numpy as np
import colorsys

def generate_watermark_frame(hue):
    diameter = 100  # 水印直径
    max_inner_rect_size = diameter * 0.7071  # 圆的最大内接正方形边长（diameter/sqrt(2)）

    # 创建一个白色的圆形背景
    image = Image.new('RGBA', (diameter, diameter), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 绘制白色背景的圆形遮罩
    draw.ellipse([0, 0, diameter-1, diameter-1], fill=(255, 255, 255, 255))

    # 设置 D 和 S 的颜色为固定值：D 绿色，S 黄色
    d_color = (0, 255, 0, 255)  # 绿色
    s_color = (255, 0, 0, 255)  # 红色
    #(255, 255, 0, 255)  # 黄色

    # 使用默认字体，并尝试加载一个更大更合适的字体
    try:
        font = ImageFont.truetype("arial.ttf", size=int(max_inner_rect_size*1.1))  # 尝试使用较大的字体大小
    except IOError:
        font = ImageFont.load_default(size=int(max_inner_rect_size*1.1))

    # 获取文本边界框以计算文本位置
    bbox_d = draw.textbbox((0, 0), "D", font=font)
    bbox_s = draw.textbbox((0, 0), "S", font=font)
    text_width_d = bbox_d[2] - bbox_d[0]
    text_height_d = bbox_d[3] - bbox_d[1]
    text_width_s = bbox_s[2] - bbox_s[0]
    text_height_s = bbox_s[3] - bbox_s[1]

    # 计算文本起始位置使 "DS" 居中
    x = (diameter - max(text_width_d, text_width_s)) / 2
    y = (diameter - max(text_height_d, text_height_s)) / 2
    
    # 绘制 "D"
    draw.text((x, y), "D", fill=d_color, font=font)
    # 绘制 "S" 叠放在 "D" 上
    draw.text((x + text_width_d/2 - text_width_s/2, y), "S", fill=s_color, font=font)

    # 创建一个与image相同尺寸的透明图层作为遮罩
    mask = Image.new('L', image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0) + image.size, fill=255)

    # 应用遮罩，确保只有圆形区域是可见的
    output = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)

    return output


def create_animated_watermark(duration, fps=20):
    frames = []
    for t in range(int(duration * fps)):
        hue = t / (duration * fps)  # 颜色随时间逐渐变化
        frame = generate_watermark_frame(hue)
        frame.save('watermark.png')
        # 将 PIL 图像转换为 NumPy 数组，并将其添加到帧列表中
        frame_np = np.array(frame)
        frames.append(frame_np)

    return ImageSequenceClip(frames, durations=[1/fps]*len(frames))

def create_floating_watermarks(video_clip, num_watermarks=10):
    video_width, video_height = video_clip.size
    watermarks = []

    animated_watermark = create_animated_watermark(video_clip.duration)

    for _ in range(num_watermarks):
        start_x = random.randint(0, video_width - animated_watermark.size[0])
        start_y = random.randint(0, video_height - animated_watermark.size[1])
        speed_x = random.uniform(-10, 10) * video_clip.fps * 1
        speed_y = random.uniform(-10, 10) * video_clip.fps * 1

        def make_frame(t):
            nonlocal start_x, start_y, speed_x, speed_y
            x = int(start_x + speed_x * t) % video_width
            y = int(start_y + speed_y * t) % video_height
            return (x, y)

        watermark = animated_watermark.set_position(make_frame)
        watermarks.append(watermark)

    return watermarks

def add_watermark(input_file, output_file):
    video = VideoFileClip(input_file)
    floating_watermarks = create_floating_watermarks(video)
    
    final_video = CompositeVideoClip([video] + floating_watermarks)
    final_video.write_videofile(output_file, codec='libx264', fps=video.fps)

if __name__ == "__main__":
    input_file = "Ranking_of_Various_Kinds_of_Books.mp4"
    output_file = "Water_Ranking_of_Various_Kinds_of_Books.mp4"
    add_watermark(input_file, output_file)