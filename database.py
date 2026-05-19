import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'escalator.db')


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'operator',
        created_at TEXT DEFAULT (datetime('now','localtime'))
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT (datetime('now','localtime')),
        video_source TEXT,
        track_id INTEGER,
        action TEXT,
        action_cn TEXT,
        confidence REAL,
        frame_idx INTEGER
    )''')
    c.execute("SELECT COUNT(*) FROM users WHERE username='admin'")
    if c.fetchone()[0] == 0:
        pw_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                  ('admin', pw_hash, 'admin'))
    conn.commit()
    conn.close()


def register_user(username, password):
    conn = _connect()
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                  (username, pw_hash))
        conn.commit()
        conn.close()
        return True, '注册成功'
    except sqlite3.IntegrityError:
        conn.close()
        return False, '用户名已存在'


def verify_login(username, password):
    conn = _connect()
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT role FROM users WHERE username=? AND password_hash=?",
              (username, pw_hash))
    row = c.fetchone()
    conn.close()
    if row:
        return True, row[0]
    return False, None


def log_alert(video_source, track_id, action, action_cn, confidence, frame_idx):
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT INTO alerts (video_source, track_id, action, action_cn, confidence, frame_idx) "
              "VALUES (?,?,?,?,?,?)",
              (video_source, track_id, action, action_cn, confidence, frame_idx))
    conn.commit()
    conn.close()


def get_recent_alerts(limit=50):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT timestamp, track_id, action_cn, confidence FROM alerts "
              "ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_alert_stats(video_source=None):
    conn = _connect()
    c = conn.cursor()
    if video_source:
        c.execute("SELECT action, action_cn, COUNT(*) FROM alerts WHERE video_source=? "
                  "GROUP BY action ORDER BY COUNT(*) DESC", (video_source,))
    else:
        c.execute("SELECT action, action_cn, COUNT(*) FROM alerts "
                  "GROUP BY action ORDER BY COUNT(*) DESC")
    rows = c.fetchall()
    conn.close()
    return rows


if __name__ == '__main__':
    init_db()
    print("数据库初始化完成")
    ok, role = verify_login('admin', 'admin123')
    print(f"admin 登录测试: {'成功' if ok else '失败'}, 角色: {role}")
    ok, msg = register_user('test', '123456')
    print(f"注册测试: {msg}")
