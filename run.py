# -*- coding: utf-8 -*-
"""
酒店异常人员实时监控识别系统 - 快速启动脚本 run.py
==================================================
功能：
  1. 检查系统环境与依赖
  2. 自动初始化数据库
  3. 启动 Flask Web 服务
  4. 支持命令行参数配置
"""

import os
import sys
import time
import argparse
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 60)
    print("  酒店异常人员实时监控识别系统 - 环境检查")
    print("=" * 60 + "\n")

    checks = [
        ("Python 版本", lambda: sys.version),
        ("工作目录", lambda: str(PROJECT_ROOT)),
    ]

    modules = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
    ]

    all_ok = True
    for module, display_name in modules:
        try:
            __import__(module)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} [未安装]")
            all_ok = False

    return all_ok


def check_mysql(host, port, user, password, database):
    """检查 MySQL 连接"""
    print("\n[数据库] 检查 MySQL 连接...")
    try:
        import pymysql
        conn = pymysql.connect(
            host=host, port=port, user=user,
            password=password, database=database,
            connect_timeout=5
        )
        conn.close()
        print(f"  ✓ MySQL 连接成功 ({host}:{port}/{database})")
        return True
    except Exception as e:
        print(f"  ⚠ MySQL 连接失败: {e}")
        print("    提示: 系统将以演示模式运行（无数据库）")
        return False


def init_demo_data():
    """初始化演示数据"""
    print("\n[数据] 检查演示数据...")

    # 检查必要目录
    dirs = [
        'system_server/templates',
        'output/alert_screenshots',
        'output/reid_train_log',
        'test_video/hotel_raw',
    ]

    for d in dirs:
        path = PROJECT_ROOT / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 创建目录: {d}")

    # 检查模板文件
    templates = ['dashboard.html', 'monitor_v2.html', 'register.html', 'alerts.html', 'persons.html']
    for t in templates:
        if (PROJECT_ROOT / 'system_server' / 'templates' / t).exists():
            print(f"  ✓ 模板: {t}")
        else:
            print(f"  ✗ 缺失模板: {t}")


def create_demo_video():
    """创建演示视频"""
    video_dir = PROJECT_ROOT / 'test_video' / 'hotel_raw'
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / 'demo.avi'

    if video_path.exists():
        print(f"  ✓ 演示视频已存在: {video_path}")
        return str(video_path)

    try:
        import numpy as np
        import cv2

        width, height = 640, 480
        fps = 15
        duration = 10  # 10秒

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 背景渐变
            frame[:, :, 0] = int(20 + 10 * np.sin(i * 0.1))
            frame[:, :, 1] = int(30 + 10 * np.cos(i * 0.1))
            frame[:, :, 2] = int(50 + 10 * np.sin(i * 0.05))

            # 模拟行人移动
            x = int(100 + 200 * np.sin(i * 0.05))
            y = int(150 + 50 * np.cos(i * 0.03))

            # 绘制"行人"框
            cv2.rectangle(frame, (x, y), (x + 40, y + 80), (200, 200, 200), 2)
            cv2.circle(frame, (x + 20, y - 5), 12, (180, 180, 200), -1)

            # 绘制第二个人
            x2 = int(400 - 150 * np.sin(i * 0.04))
            y2 = int(200 + 30 * np.sin(i * 0.06))
            cv2.rectangle(frame, (x2, y2), (x2 + 35, y2 + 70), (150, 200, 150), 2)
            cv2.circle(frame, (x2 + 17, y2 - 5), 10, (130, 180, 130), -1)

            # 添加时间戳
            cv2.putText(frame, f"Demo Frame: {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Hotel Camera 01", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            writer.write(frame)

        writer.release()
        print(f"  ✓ 已创建演示视频: {video_path}")
        return str(video_path)

    except ImportError:
        print("  ⚠ OpenCV 不可用，跳过演示视频创建")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="酒店异常人员实时监控识别系统 - 快速启动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                    # 使用默认配置启动
  python run.py --video demo.mp4    # 指定视频文件
  python run.py --port 8080        # 修改端口
  python run.py --no-browser       # 不自动打开浏览器
        """
    )

    parser.add_argument('--host', default='0.0.0.0', help='服务地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--video', default=None, help='视频文件路径')
    parser.add_argument('--rtsp', action='store_true', help='使用 RTSP 摄像头')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--no-db', action='store_true', help='不连接数据库')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    # 数据库配置
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=3306)
    parser.add_argument('--db-user', default='root')
    parser.add_argument('--db-password', default='root')
    parser.add_argument('--db-name', default='hotel_security')

    args = parser.parse_args()

    # Step 1: 环境检查
    print("\n" + "=" * 60)
    print("  酒店异常人员实时监控识别系统")
    print("  Hotel Abnormal Person Monitoring & Recognition")
    print("=" * 60)

    env_ok = check_environment()

    # Step 2: 数据库检查
    db_available = False
    if not args.no_db:
        db_available = check_mysql(
            args.db_host, args.db_port,
            args.db_user, args.db_password,
            args.db_name
        )

    # Step 3: 准备演示数据
    init_demo_data()

    # Step 4: 创建演示视频
    video_path = args.video
    if not video_path and not args.rtsp:
        video_path = create_demo_video()

    # Step 5: 构建启动命令
    cmd_parts = [
        sys.executable,
        str(PROJECT_ROOT / 'system_server' / 'app.py'),
        '--host', args.host,
        '--port', str(args.port),
        '--db_host', args.db_host,
        '--db_port', str(args.db_port),
        '--db_user', args.db_user,
        '--db_password', args.db_password,
        '--db_name', args.db_name,
    ]

    if video_path:
        cmd_parts.extend(['--video', video_path])
    elif args.rtsp:
        cmd_parts.append('--rtsp')

    if args.debug:
        cmd_parts.append('--debug')

    # Step 6: 显示启动信息
    print("\n" + "=" * 60)
    print("  系统启动配置")
    print("=" * 60)
    print(f"  服务地址: http://{args.host}:{args.port}")
    print(f"  视频源:   {video_path or ('RTSP 摄像头' if args.rtsp else '本地摄像头')}")
    print(f"  数据库:   {'MySQL' if db_available else '演示模式'}")
    print(f"  调试模式: {'开启' if args.debug else '关闭'}")
    print("=" * 60 + "\n")

    # Step 7: 启动服务
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        print(f"[提示] 3 秒后自动打开浏览器: {url}")
        print("       如需禁止，请使用 --no-browser 参数\n")

        def open_browser():
            time.sleep(3)
            webbrowser.open(url)

        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    # 执行启动
    print("[启动] 正在启动 Flask 服务器...\n")
    try:
        os.execv(sys.executable, cmd_parts)
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")
        print("\n请尝试手动运行:")
        print(f"  python {' '.join(cmd_parts[1:])}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())