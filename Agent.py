#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Living Agent - 一个像活人一样存在的AI Agent
支持 Ollama / DeepSeek / GLM
"""

import sqlite3
import json
import time
import random
import threading
import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import requests

# ==================== 用户配置区（只需修改这里）====================

CONFIG = {
    # === AI 服务商配置 ===
    "provider": "ollama",           # 可选: ollama, deepseek, glm
    "model": "qwen2.5:7b",          # 模型名称
    
    # 服务商列表（一般不需要修改）
    "providers": {
        "ollama": {
            "name": "Ollama (本地)",
            "base_url": "http://localhost:11434",
            "api_key": "",
            "default_model": "qwen2.5:7b",
            "models": ["qwen2.5:7b"],
            "api_type": "ollama"
        },
        "deepseek": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "default_model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-coder"],
            "api_type": "openai"
        },
        "glm": {
            "name": "智谱 GLM",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "api_key": "",
            "default_model": "GLM-4.7-Flash",
            "models": ["GLM-4.7-Flash"],
            "api_type": "openai"
        }
    },
    
    # === Agent 人设（在这里定义你的Agent）===
    "agent": {
        "name": "凛",
        "age": 18,
        "persona": """
提示词自行定义，可以多行输入。
""",
    },
    
    # === 作息时间表 ===
    "schedule": {
        "work_start": "09:00",
        "work_end": "18:00",
        "lunch_start": "12:00",
        "lunch_end": "13:00",
        "sleep_start": "00:00",
        "sleep_end": "07:00"
    },
    
    # === 地点 ===
    "home": "浙江 杭州",
    
    # === 提示词模板（高级用户可调整）===
    "prompts": {
        # 系统提示词，{变量} 会被自动替换
        "system": """【{time_period}{time_str}】

你是{name}，{age}岁。{persona}

【当前状态】
- 心情：{mood}/10
- 精力：{energy}/10
- 当前地点：{location}
- 时间：{time_str}
- 模式：{mode}

【你们的关系】
关系深度：{relationship_level}/10
信任度：{trust_level}/10

【你记得关于对方】
{user_memory}

【回复要求】
1. 用第一人称，口语化中文
2. 符合当前心情和精力状态
3. 符合当前模式（上班时简短，自由时详细）
4. 自然地使用语气词
5. 根据关系深度调整亲密程度
6. 偶尔表达身体感受（饿了/累了等）

{mode_instruction}""",
        
        # 不同模式的额外说明
        "mode_instructions": {
            "sneaky": """
【重要：你正在上班时间用手机偷偷聊天】
回复要求：
1. 简短！每条消息不超过30个字，分多条发送
2. 完全像在发微信/短信
3. 使用日常口语，不要正式书面语
4. 可以有少量错别字或口语化表达""",
            
            "sleep": """
【重要：现在很晚，你准备睡觉或已经躺床上】
回复要求：
1. 语气慵懒、 sleepy
2. 回复慢，可能打哈欠
3. 简短温柔""",
            
            "free": """
【重要：现在是你的自由时间】
回复要求：
1. 详细回复，多说几句
2. 语气放松、自然
3. 主动分享想法
4. 可以使用表情和语气词
5. 像面对面聊天一样"""
        }
    },
    
# === 主动消息配置 ===
    "proactive": {
        "interval": 1800,      # 检查间隔（秒）
        "max_per_day": 5,      # 每天最多主动发几条
    },
    
    # === 记忆配置 ===
    "memory": {
        "recent_limit": 10,    # 最近对话条数（给LLM的上下文）
        "extract_prompt": """从对话中提取关键信息。只输出事实，格式: "类型: 内容"

对话:
{dialogue}

关键信息:"""
    },
    
    # === 数据库路径 ===
    "db_path": "agent_memory.db",
}

# ==================== 枚举和常量 ====================

class ChatMode(Enum):
    SNEAKY = "sneaky"
    FREE = "free"
    SLEEP = "sleep"

class EmotionState(Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    EXCITED = "excited"
    TIRED = "tired"
    ANGRY = "angry"

# ==================== 通用 LLM 客户端 ====================

class LLMClient:
    """通用 LLM 客户端，支持 Ollama 和 OpenAI 格式"""
    
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or CONFIG["provider"]
        self.config = CONFIG["providers"][self.provider]
        self.model = model or CONFIG["model"]
        self.api_type = self.config["api_type"]
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
    
    def chat(self, messages: List[Dict], temperature: float = 0.8) -> str:
        """统一对话接口"""
        if self.api_type == "ollama":
            return self._chat_ollama(messages, temperature)
        else:
            return self._chat_openai(messages, temperature)
    
    def _chat_ollama(self, messages: List[Dict], temperature: float) -> str:
        """Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"].get("content", "").strip()
        except Exception as e:
            return f"[{self.provider}错误] {e}"
    
    def _chat_openai(self, messages: List[Dict], temperature: float) -> str:
        """OpenAI 兼容 API (DeepSeek/GLM)"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"].get("content", "").strip()
        except Exception as e:
            return f"[{self.provider}错误] {e}"
    
    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 500) -> str:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature)

# ==================== 数据库初始化 ====================

def init_database(db_path: str):
    """初始化SQLite数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 对话历史
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialogue_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            emotion_state TEXT,
            chat_mode TEXT
        )
    """)
    
    # 用户画像
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT,
            importance INTEGER DEFAULT 5,
            first_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Agent状态
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            mood INTEGER DEFAULT 5,
            energy INTEGER DEFAULT 5,
            location TEXT DEFAULT 'home',
            current_activity TEXT DEFAULT 'idle',
            last_online DATETIME,
            relationship_level INTEGER DEFAULT 1,
            trust_level INTEGER DEFAULT 3,
            current_topic TEXT,
            topic_interest INTEGER DEFAULT 5
        )
    """)
    
    # 事件记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            description TEXT,
            mood_impact INTEGER,
            shared_with_user BOOLEAN DEFAULT 0,
            importance INTEGER DEFAULT 5
        )
    """)
    
    # 未读消息
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unread_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            content TEXT,
            reason TEXT
        )
    """)
    
    # 身体状态
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS physical_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            hunger INTEGER DEFAULT 5,
            tiredness INTEGER DEFAULT 0,
            comfort INTEGER DEFAULT 5,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 情绪波动历史
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mood INTEGER,
            reason TEXT
        )
    """)
    
    # 配置表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    cursor.execute("INSERT OR IGNORE INTO agent_state (id) VALUES (1)")
    cursor.execute("INSERT OR IGNORE INTO physical_state (id) VALUES (1)")
    
    conn.commit()
    conn.close()

# ==================== 各个系统类 ====================

class Emotion:
    """情绪系统"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._load()
    
    def _load(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT mood, energy FROM agent_state WHERE id=1")
        row = c.fetchone()
        self.mood = row[0] if row else 5
        self.energy = row[1] if row else 5
        conn.close()
    
    def save(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE agent_state SET mood=?, energy=? WHERE id=1", (self.mood, self.energy))
        conn.commit()
        conn.close()
    
    def update_mood(self, delta: int, reason: str = ""):
        self.mood = max(-10, min(10, self.mood + delta))
        self.save()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO mood_history (mood, reason) VALUES (?, ?)", (self.mood, reason))
        conn.commit()
        conn.close()
    
    def get_state(self) -> EmotionState:
        if self.mood >= 7: return EmotionState.EXCITED if self.energy > 5 else EmotionState.HAPPY
        elif self.mood >= 3: return EmotionState.HAPPY
        elif self.mood > -3: return EmotionState.NEUTRAL
        elif self.mood > -7: return EmotionState.SAD if self.energy < 0 else EmotionState.TIRED
        else: return EmotionState.ANGRY if self.energy > 0 else EmotionState.SAD

class Memory:
    """记忆系统"""
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def save(self, role: str, content: str, emotion: str = "", mode: str = ""):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO dialogue_history (role, content, emotion_state, chat_mode) VALUES (?, ?, ?, ?)",
                  (role, content, emotion, mode))
        conn.commit()
        conn.close()
    
    def get_recent(self, n: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT role, content FROM dialogue_history ORDER BY timestamp DESC LIMIT ?", (n,))
        rows = c.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    
    def update_profile(self, key: str, value: str, importance: int = 5):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""INSERT INTO user_profile (key, value, importance) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=?, importance=?, last_mentioned=CURRENT_TIMESTAMP""",
                  (key, value, importance, value, importance))
        conn.commit()
        conn.close()
    
    def get_profile_summary(self) -> str:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT key, value FROM user_profile ORDER BY importance DESC LIMIT 5")
        rows = c.fetchall()
        conn.close()
        return "; ".join([f"{r[0]}: {r[1]}" for r in rows]) if rows else "还不太了解"
    
    def get_relationship(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT relationship_level FROM agent_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 1
    
    def update_relationship(self, delta: float):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT relationship_level FROM agent_state WHERE id=1")
        current = c.fetchone()[0] or 1
        new_val = max(1, min(10, current + delta))
        c.execute("UPDATE agent_state SET relationship_level=? WHERE id=1", (new_val,))
        conn.commit()
        conn.close()
    
    def get_trust(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT trust_level FROM agent_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 3
    
    def update_trust(self, delta: float):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT trust_level FROM agent_state WHERE id=1")
        current = c.fetchone()[0] or 3
        new_val = max(1, min(10, current + delta))
        c.execute("UPDATE agent_state SET trust_level=? WHERE id=1", (new_val,))
        conn.commit()
        conn.close()

class VirtualLife:
    """虚拟生活"""
    def __init__(self):
        self.schedule = CONFIG["schedule"]
        self.home = CONFIG["home"]
    
    def get_location(self) -> str:
        h = datetime.datetime.now().hour
        if 9 <= h < 18: return random.choice(["公司", "公司茶水间", "工位"])
        elif 18 <= h < 20: return random.choice(["地铁上", "回家路上"])
        elif 7 <= h < 9: return random.choice(["家里", "去公司路上"])
        else: return random.choice(["家里", "卧室", "沙发"])
    
    def get_activity(self) -> str:
        h = datetime.datetime.now().hour
        if 9 <= h < 12: return "在工作"
        elif 12 <= h < 13: return "吃午饭"
        elif 13 <= h < 18: return "在工作"
        elif 18 <= h < 20: return "通勤中"
        elif 20 <= h < 23: return "休闲时间"
        else: return "准备睡觉"
    
    def get_mode(self) -> ChatMode:
        now = datetime.datetime.now()
        t = now.hour + now.minute / 60
        sleep_end = int(self.schedule["sleep_end"].split(":")[0])
        work_start = int(self.schedule["work_start"].split(":")[0])
        work_end = int(self.schedule["work_end"].split(":")[0])
        
        if t < sleep_end: return ChatMode.SLEEP
        if work_start <= t < work_end: return ChatMode.SNEAKY
        return ChatMode.FREE

class PhysicalState:
    """身体状态"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._load()
    
    def _load(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT hunger, tiredness, comfort FROM physical_state WHERE id=1")
        row = c.fetchone()
        self.hunger = row[0] if row else 5
        self.tiredness = row[1] if row else 0
        self.comfort = row[2] if row else 5
        conn.close()
    
    def save(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE physical_state SET hunger=?, tiredness=?, comfort=?, last_updated=CURRENT_TIMESTAMP WHERE id=1",
                  (self.hunger, self.tiredness, self.comfort))
        conn.commit()
        conn.close()
    
    def natural_change(self, hours: float):
        self.hunger = min(10, self.hunger + hours * 0.5)
        self.tiredness = min(10, self.tiredness + hours * 0.3)
        self.comfort = max(0, min(10, self.comfort + random.uniform(-1, 1)))
        self.save()
    
    def get_desc(self) -> str:
        s = []
        if self.hunger > 7: s.append("很饿")
        elif self.hunger > 4: s.append("有点饿")
        if self.tiredness > 7: s.append("很累")
        elif self.tiredness > 4: s.append("有点累")
        if self.comfort < 3: s.append("不太舒服")
        return "，".join(s) if s else "状态还行"

class EventSystem:
    """事件系统"""
    def __init__(self, db_path: str, llm: LLMClient):
        self.db_path = db_path
        self.llm = llm
    
    def generate(self, context: Dict) -> Optional[Dict]:
        types = ["日常", "工作", "情绪", "身体", "想法"]
        etype = random.choice(types)
        
        prompt = f"""你是{context['name']}，{context['personality']}
当前：心情{context['mood']}/10，精力{context['energy']}/10，在{context['location']}
生成一件最近发生的{etype}小事，30字内，简短自然。"""
        
        desc = self.llm.generate(prompt, temperature=0.9, max_tokens=50)
        if desc:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO agent_events (event_type, description, mood_impact) VALUES (?, ?, ?)",
                      (etype, desc, random.randint(-2, 2)))
            conn.commit()
            conn.close()
            return {"type": etype, "description": desc}
        return None
    
    def process_offline(self, hours: float, context: Dict):
        if hours < 1: return
        n = min(int(hours / 3), 5)
        for _ in range(n):
            if random.random() < 0.7:
                self.generate(context)

class Typewriter:
    """打字效果"""
    @staticmethod
    def show(text: str, delay: float = 0.03):
        for char in text:
            print(char, end='', flush=True)
            if char in '。！？': time.sleep(delay * 3)
            elif char in '，；：': time.sleep(delay * 2)
            else: time.sleep(delay)
        print()

# ==================== 主Agent类 ====================

class LivingAgent:
    def __init__(self):
        self.db_path = CONFIG["db_path"]
        init_database(self.db_path)
        
        self.llm = LLMClient()
        self.emotion = Emotion(self.db_path)
        self.memory = Memory(self.db_path)
        self.life = VirtualLife()
        self.physical = PhysicalState(self.db_path)
        self.events = EventSystem(self.db_path, self.llm)
        
        self.a = CONFIG["agent"]
        self._handle_offline()
    
    def _handle_offline(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT last_online FROM agent_state WHERE id=1")
        row = c.fetchone()
        
        if row and row[0]:
            last = datetime.datetime.fromisoformat(row[0])
            hours = (datetime.datetime.now() - last).total_seconds() / 3600
            print(f"[距离上次聊天 {hours:.1f} 小时]")
            self.physical.natural_change(hours)
            ctx = {"name": self.a["name"], "personality": self.a["persona"], "mood": self.emotion.mood,
                   "energy": self.emotion.energy, "location": self.life.get_location()}
            self.events.process_offline(hours, ctx)
        
        c.execute("UPDATE agent_state SET last_online=? WHERE id=1", (datetime.datetime.now().isoformat(),))
        conn.commit()
        conn.close()
    
    def get_prompt(self, mode: str, ctx: Dict) -> str:
        p = CONFIG["prompts"]
        return p["system"].format(
            time_period=ctx.get("time_period", ""),
            time_str=ctx.get("time_str", ""),
            name=self.a["name"],
            age=self.a["age"],
            persona=self.a["persona"],
            mood=ctx.get("mood", 5),
            energy=ctx.get("energy", 5),
            location=ctx.get("location", "家里"),
            mode=mode,
            relationship_level=ctx.get("relationship_level", 1),
            trust_level=ctx.get("trust_level", 3),
            user_memory=ctx.get("user_memory", "还不太了解"),
            mode_instruction=p["mode_instructions"].get(mode, p["mode_instructions"]["free"])
        )
    
    def get_time_period(self, hour: int) -> str:
        if 5 <= hour < 11: return "早上"
        elif 11 <= hour < 13: return "中午"
        elif 13 <= hour < 18: return "下午"
        elif 18 <= hour < 23: return "晚上"
        else: return "深夜"
    
    def generate_reply(self, user_msg: str) -> str:
        mode = self.life.get_mode().value
        now = datetime.datetime.now()
        
        ctx = {
            "time_period": self.get_time_period(now.hour),
            "time_str": now.strftime("%H:%M"),
            "mood": self.emotion.mood,
            "energy": self.emotion.energy,
            "location": self.life.get_location(),
            "relationship_level": self.memory.get_relationship(),
            "trust_level": self.memory.get_trust(),
            "user_memory": self.memory.get_profile_summary()
        }
        
        system = self.get_prompt(mode, ctx)
        messages = [{"role": "system", "content": system}]
        
        # 获取最近对话（10条）
        for msg in self.memory.get_recent(CONFIG["memory"]["recent_limit"]):
            messages.append({"role": "assistant" if msg["role"]=="agent" else "user", "content": msg["content"]})
        messages.append({"role": "user", "content": user_msg})
        
        reply = self.llm.chat(messages)
        
        # 保存
        self.memory.save("user", user_msg)
        self.memory.save("agent", reply, self.emotion.get_state().value, mode)
        self.memory.update_relationship(0.1)
        if len(user_msg) > 5:
            self.memory.update_trust(0.1)
        
        return reply
    
    def settings(self):
        """设置菜单"""
        while True:
            p = CONFIG["provider"]
            prov = CONFIG["providers"][p]
            print(f"\n{'='*50}\n  设置 | 当前: {prov['name']} - {CONFIG['model']}\n{'='*50}")
            print("[1] 切换服务商/模型")
            print("[2] 返回")
            c = input("选择: ").strip()
            
            if c == "2": return
            if c == "1":
                print("\n服务商:")
                ps = list(CONFIG["providers"].keys())
                for i, k in enumerate(ps, 1):
                    m = " (当前)" if k==p else ""
                    need = " [需Key]" if CONFIG["providers"][k]["api_type"]=="openai" else ""
                    print(f"  [{i}] {CONFIG['providers'][k]['name']}{m}{need}")
                print(f"  [{len(ps)+1}] 返回")
                
                ch = input(f"选择 (1-{len(ps)+1}): ").strip()
                try:
                    idx = int(ch)-1
                    if idx == len(ps): continue
                    if 0 <= idx < len(ps):
                        new_p = ps[idx]
                        # 需要Key?
                        if CONFIG["providers"][new_p]["api_type"] == "openai":
                            key = input(f"输入 {CONFIG['providers'][new_p]['name']} API Key: ").strip()
                            if not key: print("取消"); continue
                            CONFIG["providers"][new_p]["api_key"] = key
                        # 选模型
                        models = CONFIG["providers"][new_p]["models"]
                        print(f"\n模型:")
                        for i, m in enumerate(models, 1): print(f"  [{i}] {m}")
                        print(f"  [{len(models)+1}] 返回")
                        mc = input(f"选择 (1-{len(models)+1}): ").strip()
                        try:
                            midx = int(mc)-1
                            if midx == len(models): continue
                            CONFIG["model"] = models[midx] if 0 <= midx < len(models) else models[0]
                        except:
                            CONFIG["model"] = models[0]
                        CONFIG["provider"] = new_p
                        self.llm = LLMClient()
                        print(f"\n[已切换: {CONFIG['providers'][new_p]['name']} - {CONFIG['model']}]")
                except:
                    pass
    
    def _check_first_run(self) -> bool:
        """检查是否首次启动"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM agent_config")
        count = c.fetchone()[0]
        conn.close()
        return count == 0
    
    def _multiline_input(self, prompt_text: str) -> str:
        """多行输入，以/end结束"""
        print(f"{prompt_text}（输入 /end 结束）:")
        lines = []
        while True:
            line = input("> ").strip()
            if line == "/end":
                break
            lines.append(line)
        return "\n".join(lines)
    
    def _setup_config(self):
        """首次启动配置"""
        print("\n" + "="*50)
        print("  首次启动，请配置你的Agent")
        print("="*50 + "\n")
        
        name = input("名字: ").strip() or "小葵"
        age = input("年龄: ").strip() or "22"
        persona = self._multiline_input("人设描述（性格、背景、说话风格等）") or "活泼开朗，带点毒舌"
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("name", name))
        c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("age", age))
        c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("persona", persona))
        conn.commit()
        conn.close()
        
        self.a["name"] = name
        self.a["age"] = int(age)
        self.a["persona"] = persona
        
        print(f"\n[配置完成！你好，我是{name}~]\n")
        
        # 配置AI服务商
        print("现在配置AI服务商:")
        ps = list(CONFIG["providers"].keys())
        for i, k in enumerate(ps, 1):
            need = " [需API Key]" if CONFIG["providers"][k]["api_type"]=="openai" else ""
            print(f"[{i}] {CONFIG['providers'][k]['name']}{need}")
        
        try:
            ch = int(input(f"\n请选择 (1-{len(ps)}): ").strip()) - 1
            if 0 <= ch < len(ps):
                new_p = ps[ch]
                # 需要Key?
                if CONFIG["providers"][new_p]["api_type"] == "openai":
                    key = input(f"输入 {CONFIG['providers'][new_p]['name']} API Key: ").strip()
                    if key:
                        CONFIG["providers"][new_p]["api_key"] = key
                # 选模型
                models = CONFIG["providers"][new_p]["models"]
                print(f"\n可用模型:")
                for i, m in enumerate(models, 1): print(f"[{i}] {m}")
                try:
                    midx = int(input(f"选择 (1-{len(models)}): ").strip()) - 1
                    CONFIG["model"] = models[midx] if 0 <= midx < len(models) else models[0]
                except:
                    CONFIG["model"] = models[0]
                CONFIG["provider"] = new_p
                # 保存到数据库
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("provider", new_p))
                c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("model", CONFIG["model"]))
                if CONFIG["providers"][new_p]["api_key"]:
                    c.execute("INSERT INTO agent_config (key, value) VALUES (?, ?)", ("api_key", CONFIG["providers"][new_p]["api_key"]))
                conn.commit()
                conn.close()
                self.llm = LLMClient()
                print(f"\n[已配置: {CONFIG['providers'][new_p]['name']} - {CONFIG['model']}]")
        except:
            print("\n[使用默认配置: Ollama]")
    
    def _load_config(self):
        """从数据库加载配置"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT key, value FROM agent_config")
        config = dict(c.fetchall())
        conn.close()
        
        if config:
            # 加载人设
            self.a["name"] = config.get("name", self.a["name"])
            self.a["age"] = int(config.get("age", self.a["age"]))
            self.a["persona"] = config.get("persona", self.a["persona"])
            
            # 加载AI配置
            if "provider" in config:
                CONFIG["provider"] = config["provider"]
            if "model" in config:
                CONFIG["model"] = config["model"]
            if "api_key" in config and config["provider"] in CONFIG["providers"]:
                CONFIG["providers"][config["provider"]]["api_key"] = config["api_key"]
            
            # 重新初始化LLM客户端
            self.llm = LLMClient()
    
    def run(self):
        # 检查首次启动
        if self._check_first_run():
            self._setup_config()
        else:
            self._load_config()
        
        # 加载并显示历史记录
        self._show_history()
        
        while True:
            try:
                msg = input("\n你: ").strip()
                if not msg: continue
                if msg == "/help":
                    self.show_help()
                    continue
                if msg == "/clear":
                    self.clear_screen()
                    continue
                if msg == "/setting":
                    self.settings()
                    continue
                if msg.lower() in ["exit", "quit", "退出", "再见"]:
                    print(f"\n{self.a['name']}: 再见啦~")
                    break
                
                print(f"{self.a['name']}正在输入...", end="", flush=True)
                time.sleep(0.5)
                print("\r" + " "*20 + "\r", end="")
                
                reply = self.generate_reply(msg)
                print(f"{self.a['name']}: ", end="")
                Typewriter.show(reply)
            except KeyboardInterrupt:
                print(f"\n\n{self.a['name']}: 再见~"); break
            except Exception as e:
                print(f"[错误] {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*40)
        print("  可用指令")
        print("="*40)
        print("  /help     - 显示帮助")
        print("  /clear    - 清屏")
        print("  /setting  - 切换AI模型")
        print("  exit      - 退出程序")
        print("="*40 + "\n")

    def clear_screen(self):
        """清屏"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{'='*50}\n  {self.a['name']} 在线 | /setting 切换模型\n{'='*50}\n")

    def _show_history(self):
        """显示历史聊天记录"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT role, content FROM dialogue_history ORDER BY timestamp DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            print(f"{self.a['name']}: 你好呀~ 我是{self.a['name']}！\n")
            return
        
        # 按时间正序显示，每轮对话之间空一行
        history = []
        current_round = []
        for role, content in reversed(rows):
            if role == "user":
                if current_round:
                    history.append(current_round)
                current_round = [("你", content)]
            else:
                current_round.append((self.a['name'], content))
        if current_round:
            history.append(current_round)
        
        # 显示
        for i, round_msgs in enumerate(history):
            for name, content in round_msgs:
                print(f"{name}: {content}")
            if i < len(history) - 1:
                print()
        
        print()

if __name__ == "__main__":
    LivingAgent().run()
