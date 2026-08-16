# -*- coding: utf-8 -*-
"""
数据库交互模块 db_mysql.py
封装 hotel_security 数据库的全部增删改查操作
"""

import os
import json
import threading
import pymysql
import pymysql.cursors
from datetime import datetime
from typing import List, Dict, Optional, Any


class HotelDatabase:
    """酒店安防数据库操作类"""

    def __init__(self, host='localhost', port=3306,
                 user='root', password='123456',
                 database='hotel_security', charset='utf8mb4'):
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': charset,
            'cursorclass': pymysql.cursors.DictCursor,
            # MySQL 8 默认 caching_sha2_password; 禁用 SSL 后用 RSA 交换(需 cryptography 包),
            # 避免本地自签名证书导致的 ASN1 SSL 握手失败
            'ssl_disabled': True
        }
        self._conn = None
        # pymysql 连接非线程安全; 全局 db 单例被 Flask 多线程并发访问时,
        # 会导致 "Packet sequence number wrong" / "read of closed file"。
        # 用锁串行化同一实例的 DB 操作 (不同实例各自独立连接, 互不影响)。
        self._lock = threading.Lock()

    def _get_connection(self):
        """获取数据库连接"""
        if self._conn is None or not self._conn.open:
            self._conn = pymysql.connect(**self.config)
        return self._conn

    def _execute(self, sql, params=None, fetch=True):
        """执行SQL语句"""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    if fetch:
                        return cursor.fetchall()
                    conn.commit()
                    return cursor.rowcount
            except Exception as e:
                conn.rollback()
                raise e

    def _execute_insert(self, sql, params=None):
        """执行 INSERT 并返回自增主键 ID"""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    conn.commit()
                    return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                raise e

    # ========================================================
    # person_info 表操作
    # ========================================================

    def insert_person(self, name: str, id_card: str, room_num: str,
                      check_in_time: Optional[str] = None,
                      feature_vec: Optional[str] = None,
                      face_vec: Optional[str] = None,
                      face_img_path: Optional[str] = None) -> int:
        """
        插入新住客信息
        :return: 新记录ID
        """
        if check_in_time is None:
            check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        sql = """
        INSERT INTO person_info (name, id_card, room_num, check_in_time, feature_vec, face_vec, face_img_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (name, id_card, room_num, check_in_time, feature_vec, face_vec, face_img_path)
        return self._execute_insert(sql, params)

    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """根据ID查询住客"""
        sql = "SELECT * FROM person_info WHERE id = %s"
        results = self._execute(sql, (person_id,))
        return results[0] if results else None

    def get_person_by_id_card(self, id_card: str) -> Optional[Dict]:
        """根据身份证查询住客"""
        sql = "SELECT * FROM person_info WHERE id_card = %s"
        results = self._execute(sql, (id_card,))
        return results[0] if results else None

    def get_all_persons(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """查询所有住客"""
        sql = "SELECT * FROM person_info ORDER BY id DESC LIMIT %s OFFSET %s"
        return self._execute(sql, (limit, offset))

    def update_person(self, person_id: int, **kwargs) -> bool:
        """更新住客信息"""
        allowed_fields = {'name', 'id_card', 'room_num', 'check_in_time',
                          'check_out_time', 'feature_vec', 'face_vec', 'face_img_path'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return False

        set_clause = ', '.join([f"{k} = %s" for k in updates.keys()])
        params = list(updates.values()) + [person_id]

        sql = f"UPDATE person_info SET {set_clause} WHERE id = %s"
        return self._execute(sql, params, fetch=False) > 0

    def delete_person(self, person_id: int) -> bool:
        """删除住客"""
        sql = "DELETE FROM person_info WHERE id = %s"
        return self._execute(sql, (person_id,), fetch=False) > 0

    def search_persons(self, keyword: str) -> List[Dict]:
        """搜索住客(按姓名/身份证/房号)"""
        sql = """
        SELECT * FROM person_info
        WHERE name LIKE %s OR id_card LIKE %s OR room_num LIKE %s
        ORDER BY id DESC
        """
        kw = f"%{keyword}%"
        return self._execute(sql, (kw, kw, kw))

    def update_feature(self, person_id: int, feature_vec: str) -> bool:
        """更新特征向量"""
        sql = "UPDATE person_info SET feature_vec = %s WHERE id = %s"
        return self._execute(sql, (feature_vec, person_id), fetch=False) > 0

    def update_face_vec(self, person_id: int, face_vec: str) -> bool:
        """更新人脸特征向量 (512维)"""
        sql = "UPDATE person_info SET face_vec = %s WHERE id = %s"
        return self._execute(sql, (face_vec, person_id), fetch=False) > 0

    # ========================================================
    # alert_log 表操作
    # ========================================================

    def insert_alert(self, camera_id: str, screenshot_path: str,
                     similarity: float, handle_status: int = 0,
                     person_key: Optional[str] = None,
                     feature_vec: Optional[str] = None,
                     embedding_type: Optional[str] = None) -> int:
        """
        插入预警记录
        :return: 新记录ID
        """
        sql = """
        INSERT INTO alert_log (alert_time, camera_id, screenshot_path, similarity, handle_status,
                               person_key, feature_vec, embedding_type)
        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
        """
        return self._execute_insert(sql, (camera_id, screenshot_path, similarity, handle_status,
                                          person_key, feature_vec, embedding_type))

    def get_alert_by_id(self, log_id: int) -> Optional[Dict]:
        """根据ID查询预警"""
        sql = "SELECT * FROM alert_log WHERE log_id = %s"
        results = self._execute(sql, (log_id,))
        return results[0] if results else None

    def get_all_alerts(self, limit: int = 100, offset: int = 0,
                        status: Optional[int] = None) -> List[Dict]:
        """查询预警列表"""
        if status is not None:
            sql = "SELECT * FROM alert_log WHERE handle_status = %s ORDER BY alert_time DESC LIMIT %s OFFSET %s"
            return self._execute(sql, (status, limit, offset))
        else:
            sql = "SELECT * FROM alert_log ORDER BY alert_time DESC LIMIT %s OFFSET %s"
            return self._execute(sql, (limit, offset))

    def get_alerts_by_camera(self, camera_id: str, limit: int = 50) -> List[Dict]:
        """按摄像头查询预警"""
        sql = "SELECT * FROM alert_log WHERE camera_id = %s ORDER BY alert_time DESC LIMIT %s"
        return self._execute(sql, (camera_id, limit))

    def update_alert_status(self, log_id: int, handle_status: int) -> bool:
        """更新预警处理状态"""
        sql = "UPDATE alert_log SET handle_status = %s WHERE log_id = %s"
        return self._execute(sql, (handle_status, log_id), fetch=False) > 0

    def delete_alert(self, log_id: int) -> bool:
        """删除预警记录"""
        sql = "DELETE FROM alert_log WHERE log_id = %s"
        return self._execute(sql, (log_id,), fetch=False) > 0

    def delete_all_alerts(self) -> int:
        """一键清空所有预警记录, 返回删除行数"""
        return self._execute("DELETE FROM alert_log", fetch=False)

    def get_alerts_by_person_key(self, person_key: str) -> List[Dict]:
        """查询同一异常人员 (person_key) 的全部预警截图记录"""
        sql = "SELECT * FROM alert_log WHERE person_key = %s ORDER BY alert_time ASC"
        return self._execute(sql, (person_key,))

    def delete_alerts_by_person_key(self, person_key: str) -> bool:
        """删除同一异常人员的全部预警记录"""
        sql = "DELETE FROM alert_log WHERE person_key = %s"
        return self._execute(sql, (person_key,), fetch=False) > 0

    def mark_alerts_handled_by_person_key(self, person_key: str) -> bool:
        """标记同一异常人员的全部预警为已处置"""
        sql = "UPDATE alert_log SET handle_status = 1 WHERE person_key = %s"
        return self._execute(sql, (person_key,), fetch=False) > 0

    def get_unhandled_count(self) -> int:
        """获取未处理预警数量"""
        sql = "SELECT COUNT(*) as cnt FROM alert_log WHERE handle_status = 0"
        results = self._execute(sql)
        return results[0]['cnt'] if results else 0

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计数据"""
        stats = {}

        # 住客统计
        sql_total = "SELECT COUNT(*) as cnt FROM person_info"
        stats['total_persons'] = self._execute(sql_total)[0]['cnt']

        # 预警统计
        sql_alerts = """
        SELECT
            COUNT(*) as total_alerts,
            SUM(CASE WHEN handle_status = 0 THEN 1 ELSE 0 END) as unhandled,
            SUM(CASE WHEN handle_status = 1 THEN 1 ELSE 0 END) as handled
        FROM alert_log
        """
        alert_stats = self._execute(sql_alerts)[0]
        stats['total_alerts'] = alert_stats['total_alerts']
        stats['unhandled_alerts'] = alert_stats['unhandled']
        stats['handled_alerts'] = alert_stats['handled']

        # 今日预警
        sql_today = "SELECT COUNT(*) as cnt FROM alert_log WHERE DATE(alert_time) = CURDATE()"
        stats['today_alerts'] = self._execute(sql_today)[0]['cnt']

        return stats

    # ========================================================
    # 数据库管理
    # ========================================================

    def close(self):
        """关闭数据库连接"""
        if self._conn and self._conn.open:
            self._conn.close()
            self._conn = None

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            conn = self._get_connection()
            conn.ping()
            return True
        except Exception as e:
            print(f"[ERROR] 数据库连接失败: {e}")
            return False

    def ensure_alert_schema(self) -> bool:
        """
        幂等迁移: 确保 alert_log 包含按人聚合所需列与索引。
        MySQL 8 不支持 ADD COLUMN IF NOT EXISTS, 故先查 information_schema 再决定是否 ALTER。
        """
        try:
            cols = {r['COLUMN_NAME'] for r in self._execute(
                "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'alert_log'"
            )}
            additions = {
                'person_key': "ALTER TABLE alert_log ADD COLUMN person_key VARCHAR(64) DEFAULT NULL",
                'feature_vec': "ALTER TABLE alert_log ADD COLUMN feature_vec TEXT DEFAULT NULL",
                'embedding_type': "ALTER TABLE alert_log ADD COLUMN embedding_type VARCHAR(8) DEFAULT NULL",
            }
            for col, sql in additions.items():
                if col not in cols:
                    self._execute(sql, fetch=False)
                    print(f"[INFO] alert_log 新增列: {col}")

            idx = self._execute(
                "SELECT COUNT(*) AS cnt FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'alert_log' AND INDEX_NAME = 'idx_person_key'"
            )
            if idx and idx[0]['cnt'] == 0:
                self._execute("ALTER TABLE alert_log ADD INDEX idx_person_key (person_key)", fetch=False)
                print("[INFO] alert_log 新增索引: idx_person_key")
            return True
        except Exception as e:
            print(f"[WARN] alert_log 表迁移失败 (可能表尚未创建): {e}")
            return False