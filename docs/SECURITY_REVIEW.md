# 安全审查 — 本地 Notebook Builder v2

## 关键结论

**不能作为不可信用户 Python 执行沙箱。** Notebook 的任何 code Cell 都能够执行运行账户拥有的 Python、文件系统和网络操作；即使 MCP 访问路径被限制到 `NB_ROOT`，也不代表 Python Kernel 被限制到同一路径。公网暴露、陌生用户上传执行、未经检查的依赖安装与秘密凭据注入均不适合当前实现。

| 威胁 | 防护 | 尚存风险 |
| --- | --- | --- |
| Notebook 路径穿越 | 相对路径、`.ipynb` 类型、`NB_ROOT` 解析、隐藏目录拒绝 | 本地恶意用户可利用路径检查与后续文件操作之间的 TOCTOU 竞态 |
| 符号链接逃逸 | 工作区与历史目录 resolve 检查；锁文件 `O_NOFOLLOW` | 无强制 OS 命名空间隔离或 openat2 完整保护 |
| 非预期修改 | 默认 `NB_MCP_READONLY=1`，创建/编辑工具拦截 | 拥有 Python 直接访问权的 CLI 用户不受 MCP 门禁限制 |
| 非预期执行 | 还需要 `NB_MCP_ALLOW_EXEC=1`；默认禁止 | 一旦允许，执行可做任意同用户权限操作 |
| 并发覆盖 | `fcntl.flock` 与 `expected_version`、原子替换 | 其他编辑器不守协议时可能绕过锁 |
| 重放副作用 | 128 个最近 Cell 幂等收据；SQLite Run Idempotency Key | 收据老化后无法无限期去重；外部 Python 文件/网络副作用不能回滚 |
| 任务崩溃 | 缺失进程标记 interrupted、不自动重跑 | 文件写入与 SQLite 状态不是跨系统原子事务 |
| 资源耗尽 | Notebook 20MB、Cell 源码 2MB、工作区文件 2MB、最多 4 个 MCP Kernel、每文件最多 1 个活跃 Run | 没有 cgroup/CPU/RAM/网络限额；代码可能无限循环或耗尽机器资源 |
| 响应超大 | MCP 单入站消息 3MB，返回 2MB；富文本输出截断 | HTML 生成及 Python 执行依然受资源限制影响 |
| 数据暴露 | 变量探针只返回类型、shape/长度，不输出 repr 值 | 变量名可能泄露业务信息；Notebook 输出可能含秘密 |

## 推荐部署

1. 使用专用低权限 Unix 账户运行 MCP，不要授予 root 或 Docker socket。
2. 把 `NB_ROOT` 设置为不包含机密的专用目录，默认保持 MCP 只读。
3. 对不受信任的 Notebook **不启用执行**，必须执行时使用额外隔离 VM/容器与限制网络访问。
4. 不将 stdio MCP 包装成不受认证的公开 HTTP 服务。
5. 对敏感任务执行前复核源码、外部包安装命令、网络行为与输出落盘位置。
6. 本地日志和 Notebook 输出中的秘密不自动脱敏；对 HTML、Notebook 共享进行独立隐私审查。

## 验证覆盖

已有自动测试：路径遍历、符号链接逃逸、只读门禁、执行门禁、参数 schema、CAS 并发、历史版本、冲突选择、错误执行不覆盖原文档、幂等重放、作业取消、进程死亡转 interrupted。尚无 fuzz、进程级压力、跨平台 Windows/macOS、cgroup 基准、任意客户端版本兼容性或真实 GPU 测试。
