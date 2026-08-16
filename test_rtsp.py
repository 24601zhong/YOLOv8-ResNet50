# -*- coding: utf-8 -*-
"""
RTSP 网络摄像头连通性诊断脚本
================================
用途：在不启动整套系统的情况下，单独验证某个 RTSP 摄像头能否被 OpenCV 打开，
并给出具体失败原因（网络不通 / 端口不对 / 路径不对 / 账号密码不对 / 编码不支持）。

用法示例：
    # 方式一：只给完整地址
    python test_rtsp.py --url "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101"

    # 方式二：分字段传参（路径留空则用 /）
    python test_rtsp.py --ip 192.168.1.64 --port 554 --user admin --pass 12345 --path /Streaming/Channels/101

常用 RTSP 路径（不同品牌不一样）：
    海康 Hikvision : /Streaming/Channels/101   (主码流)   /Streaming/Channels/102 (子码流)
    大华 Dahua     : /cam/realmonitor?channel=1&subtype=0
    通用/ONVIF     : /live  或  /stream1
"""

import argparse
import socket
import sys
import os


def build_rtsp_url(ip, port=554, username=None, password=None, path=''):
    ip = str(ip).strip()
    path = str(path or '').strip()
    if path and not path.startswith('/'):
        path = '/' + path
    auth = ''
    if username:
        auth = f"{username}:{password or ''}@"
    return f"rtsp://{auth}{ip}:{port}{path}"


def check_tcp(ip, port, timeout=3):
    """第一步：TCP 端口是否可达（能快速区分'网络不通'与'流打不开'）"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="RTSP 摄像头连通性诊断")
    parser.add_argument("--url", type=str, default=None, help="完整 RTSP 地址")
    parser.add_argument("--ip", type=str, default=None, help="摄像头 IP")
    parser.add_argument("--port", type=int, default=554, help="RTSP 端口(默认554)")
    parser.add_argument("--user", type=str, default=None, help="用户名")
    parser.add_argument("--pass", dest="pwd", type=str, default=None, help="密码")
    parser.add_argument("--path", type=str, default='', help="RTSP 路径(如 /Streaming/Channels/101)")
    args = parser.parse_args()

    if args.url:
        url = args.url
        print(f"[1/3] 目标地址: {url}")
    else:
        if not args.ip:
            print("[!] 请提供 --url 或至少 --ip")
            sys.exit(1)
        url = build_rtsp_url(args.ip, args.port, args.user, args.pwd, args.path)
        print(f"[1/3] 构造地址: {url}")

    # 从 URL 里拆出 ip/port 做 TCP 检查
    host, port = None, 554
    try:
        rest = url.split('://', 1)[1]
        # 去掉 user:pass@
        if '@' in rest:
            rest = rest.split('@', 1)[1]
        hp = rest.split('/', 1)[0]
        if ':' in hp:
            host, port = hp.rsplit(':', 1)
            port = int(port)
        else:
            host = hp
    except Exception:
        host = None

    if host:
        ok, err = check_tcp(host, port)
        if not ok:
            print(f"[2/3] TCP 连接失败: {host}:{port} 不可达 -> {err}")
            print("      原因大概率是: IP 错误 / 摄像头关机 / 不在同一网段 / 防火墙拦截 / 端口错误")
            sys.exit(2)
        print(f"[2/3] TCP 连接 {host}:{port} 正常 ✓")
    else:
        print("[2/3] 跳过 TCP 检查（无法解析地址）")

    # 第二步：OpenCV FFMPEG 打开
    print("[3/3] 正在用 OpenCV FFMPEG 打开流（最多约 5 秒超时）...")
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
        'rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|probesize;1000000|analyzeduration;1000000'

    import cv2
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("\n[结果] ✗ 无法打开流。TCP 通但流打不开，通常是以下原因之一：")
        print("       1) RTSP 路径(path)不对 —— 不同品牌路径不同，见文件头部说明")
        print("       2) 账号/密码错误")
        print("       3) 该通道是 H.265 编码，当前 OpenCV 的 FFMPEG 无法解码(可改用 H.264 子码流)")
        print("       4) 摄像头未开启 RTSP 协议(需进摄像头网页后台开启)")
        cap.release()
        sys.exit(3)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("\n[结果] ✗ 流打开了但读不到画面，可能是编码不支持(H.265)或通道无视频。")
        cap.release()
        sys.exit(4)

    h, w = frame.shape[:2]
    print(f"\n[结果] ✓ 成功！读到第一帧，分辨率 {w}×{h}")
    print(f"       该地址可直接填到系统'摄像头'页面对应的 IP/端口/账号/密码/路径里。")
    cap.release()


if __name__ == '__main__':
    main()
