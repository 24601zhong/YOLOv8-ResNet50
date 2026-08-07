-- ========================================================
-- 酒店异常人员实时监控识别系统 - 数据库初始化脚本
-- 数据库版本: MySQL 8.0
-- 字符集: utf8mb4
-- ========================================================

CREATE DATABASE IF NOT EXISTS hotel_security 
  DEFAULT CHARACTER SET utf8mb4 
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE hotel_security;

-- ========================================================
-- 表1: person_info（住客信息表）
-- 存储已登记入住客人的基本信息及人脸特征向量
-- ========================================================
DROP TABLE IF EXISTS person_info;
CREATE TABLE person_info (
  id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
  name VARCHAR(50) NOT NULL COMMENT '姓名',
  id_card VARCHAR(18) NOT NULL COMMENT '身份证号',
  room_num VARCHAR(10) NOT NULL COMMENT '房间号',
  check_in_time DATETIME NOT NULL COMMENT '入住时间',
  check_out_time DATETIME DEFAULT NULL COMMENT '离店时间',
  feature_vec TEXT COMMENT '2048维行人特征向量(JSON数组存储)',
  face_img_path VARCHAR(255) COMMENT '登记抓拍图片路径',
  UNIQUE INDEX uk_id_card (id_card) COMMENT '身份证唯一索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='住客信息表';

-- ========================================================
-- 表2: alert_log（异常预警日志表）
-- 存储系统检测到的异常人员预警记录
-- ========================================================
DROP TABLE IF EXISTS alert_log;
CREATE TABLE alert_log (
  log_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
  alert_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '预警时间',
  camera_id VARCHAR(30) NOT NULL COMMENT '摄像头点位编号',
  screenshot_path VARCHAR(255) COMMENT '预警截图路径',
  similarity FLOAT COMMENT '匹配相似度(0-1)',
  handle_status TINYINT NOT NULL DEFAULT 0 COMMENT '处理状态: 0-未处理, 1-已处置'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='异常预警日志表';

-- ========================================================
-- 插入示例测试数据
-- ========================================================
INSERT INTO person_info (name, id_card, room_num, check_in_time, feature_vec, face_img_path) VALUES
('张三', '110101199001011234', '8001', '2026-08-01 14:30:00', '[0.12,-0.34,...]', 'data/face/zhangsan.jpg'),
('李四', '110101198505056789', '8002', '2026-08-02 10:15:00', '[0.56,0.78,...]', 'data/face/lisi.jpg');

INSERT INTO alert_log (alert_time, camera_id, screenshot_path, similarity, handle_status) VALUES
('2026-08-07 20:15:33', 'CAM_001', 'output/alert_screenshots/alert_001.jpg', 0.72, 0),
('2026-08-07 21:02:18', 'CAM_002', 'output/alert_screenshots/alert_002.jpg', 0.68, 1);

-- 验证语句
SELECT 'person_info表创建完成，共' AS info, COUNT(*) AS count FROM person_info
UNION ALL
SELECT 'alert_log表创建完成，共', COUNT(*) FROM alert_log;