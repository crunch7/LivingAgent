# LivingAgent

一个像活人一样存在的 AI Agent，支持 Ollama / DeepSeek / GLM。

## ✨ 特色

- 🎭 **拟人化交互**：根据时间、情绪、精力动态调整回复风格
- 🧠 **记忆系统**：SQLite 存储对话历史与用户画像
- ⏰ **作息模拟**：上班/自由/睡眠模式，不同时间段不同语气
- 💪 **情绪与身体状态**：心情和身体状态会随时间自然变化
- 🔄 **主动生活**：自动生成日常事件，离线时有事件积累
- ⚙️ **多服务商支持**：一键切换 Ollama、DeepSeek、智谱 GLM

## 📦 结构

```
LivingAgent/
├── Agent.py              # 主程序
├── agent_memory.db       # SQLite 数据库（首次运行自动创建）
├── CONFIG                # 配置区域（在 Agent.py 顶部）
└── README.md             # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests
```

### 2. 配置 AI 服务商

编辑 `Agent.py` 顶部的 `CONFIG` 部分：

```python
CONFIG = {
    "provider": "ollama",  # ollama / deepseek / glm
    "model": "qwen2.5:7b",
    # ... 其他配置
}
```

- **Ollama**：确保本地 Ollama 运行在 `http://localhost:11434`
- **DeepSeek / GLM**：填写对应 API Key

### 3. 运行

```bash
python Agent.py
```

首次运行会引导你配置 Agent 的人设（名字、年龄、性格等）。

## ⌨️ 指令

运行后可用指令：
- `/help` - 显示帮助
- `/clear` - 清屏
- `/setting` - 切换 AI 模型/服务商
- `exit` / `quit` - 退出

## 🔧 配置说明

### 人设（CONFIG["agent"]）

```python
"agent": {
    "name": "凛",      # Agent 名字
    "age": 18,         # 年龄
    "persona": "...",  # 性格描述，多行文本
}
```

### 作息时间（CONFIG["schedule"]）

```python
"schedule": {
    "work_start": "09:00",
    "work_end": "18:00",
    "lunch_start": "12:00",
    "lunch_end": "13:00",
    "sleep_start": "00:00",
    "sleep_end": "07:00"
}
```

- `work_start` ~ `work_end`：上班时间，模式切换为 **SNEAKY**（简短回复）
- 其他时间：**FREE**（自由详细回复）
- `sleep_start` ~ `sleep_end`：**SLEEP** 模式（慵懒）

### 地点（CONFIG["home"]）

默认地点：`"浙江 杭州"`

### 提示词模板（CONFIG["prompts"]）

可以自定义系统提示词，支持以下变量：
- `{name}`, `{age}`, `{persona}`
- `{time_period}`, `{time_str}`
- `{mood}`, `{energy}`
- `{location}`, `{mode}`
- `{relationship_level}`, `{trust_level}`
- `{user_memory}`
- `{mode_instruction}`

### AI 提供商（CONFIG["providers"]）

预置了三个提供商：
1. **ollama**：本地，无需 API Key
2. **deepseek**：DeepSeek API
3. **glm**：智谱 GLM API

你也可以添加新的提供商（遵循 OpenAI 兼容格式）。

## 💾 数据库说明

首次运行会自动创建 `agent_memory.db`，包含表：
- `dialogue_history`：对话历史
- `user_profile`：用户画像（提取的关键信息）
- `agent_state`：Agent 状态（心情、精力、关系度等）
- `agent_events`：事件记录
- `unread_messages`：未读消息
- `physical_state`：身体状态
- `mood_history`：情绪变化历史
- `agent_config`：持久化配置

## 🧠 工作原理

1. **启动时**：加载配置，恢复状态，处理离线时间产生的事件
2. **每次对话**：
   - 根据当前时间、状态、记忆生成系统提示词
   - 调用 LLM 生成回复
   - 保存对话到数据库
   - 更新关系度和信任度
3. **状态变化**：情绪、精力、身体状态会随时间、对话内容变化

## 📝 示例对话

```
你: 在干嘛
凛: 刚吃完饭,在工位上摸鱼~ 😴
```

（如果在上班时间，回复会简短；如果是晚上，会比较详细）

## 🔐 安全提醒

- **不要提交包含 API Key 的代码到公开仓库**
- 建议使用环境变量或本地配置文件存储敏感信息
- 数据库文件 `agent_memory.db` 包含对话记录，注意隐私

## 📄 许可

MIT License（或根据你的项目选择）

---

**享受和你的 LivingAgent 聊天吧！** 🎉
