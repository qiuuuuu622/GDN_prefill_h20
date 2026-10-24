# 来源、公开范围与复现边界

## 来源

122B内容来自本仓库原提交 `9a3aaeabf8efee7394cf46ff37f163101c72c5e9`。原有 `flashinfer/`、`patch/`、`tests/`、`bench/` 保持；原README另存为 [122B历史报告](122b-original-report.md)。公开数值目前是原报告精度，本次未取得其逐次计时日志，不补造原始记录。

35B来自2026-10-23/24的独立实验归档。每个导入文件的原相对路径、原SHA256、发布后SHA256及变换见 [source_manifest.json](../data/source_manifest.json)。核心runtime与原实测release字节一致；部分归档增加历史说明、脱敏路径或删去非数值请求字段，均记录变换。

## 已公开

- 核心代码、上游patch、源码快照、内核生成器、历史测试与控制脚本。
- 每轮报告、启动参数数组、模型/软件geometry与版本。
- 合成shape扫描、正确性JSON、在线shadow数值、捕获张量哈希。
- 逐请求计时、token数、样本编号、输出哈希、batch/缓存数值及汇总。
- 失败/拒绝候选的数值结果和报告；有性能退化的轮次没有删除。

## 未公开

`fixed10.json` payload、业务提示词、图片、SSE模型输出正文、模型权重、真实激活`.pt`、完整服务日志、完整profiler trace、PID与编译缓存。原报告中这些路径属于私有原始归档，不保证在这个公开摘录中存在。真实激活回放需要原张量；全文输出一致性重算需要原SSE。公开计时表可独立重算。

数据集SHA256：

```text
d5e8c10e433ea5952eaba44346489dd9161637bd1ed94060f45347f32717b30e
```

这是原固定十条文件的哈希，不是脱敏请求计时JSONL的哈希。每个公开文件由manifest另行记录。

## 不混用的证据

- 合成CUDA Graph设备时间、完整extend组件时间、真实张量回放与客户端TTFT分列。
- 原122B报告使用FlashInfer开发版；新增统一实验在安装版0.6.18上同时测Hq4/Hv16和Hq16/Hv32，单独列为同版本对照，不与旧表拼接。
- 35B不同实验的baseline各自重测，不将跨轮baseline拼成单个AB。
- 原122B早期bench有log-space gate导致NaN的已知问题；原报告称有效linear gate重新测量并校验。保留旧脚本用于审计，不能直接将其视为当前正确的benchmark。见历史报告Caveats。
- 35B shape-study中的失败pilot、作废组件结果和高精度阈值失败作为历史证据保留，不进入正式收益表。
- 自动profiler分类中的FA3被归到gemm、自动fuse目录匹配等只是工具输出，统一报告按全注意力解释，不作为已验证优化。

## 第三方代码

FlashInfer/SGLang源码快照及生成内核保留原版权和许可证头；新增整理不改变这些文件的权利声明。仓库历史未提供统一顶层许可证，本次不替原作者重新授权。

## 新增统一实验

统一checkpoint、CP边界修复以及公共入口验证独立归档。新core代码位于src，历史源码保持冻结。各实验内部原manifest记录源归档版本；公开摘录的路径脱敏以根data/source_manifest.json的published_sha256为准。公共入口的新写代码、CPU测试及GPU验证不是历史计时所用的hook，不将其混为同一版本。
