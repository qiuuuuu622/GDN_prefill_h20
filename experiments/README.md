# 实验演进索引

每轮目录是公开摘录，不包含私有payload、SSE、原始激活或完整服务日志。历史README保持当时结论，后续修正以根README和统一机制/验证文档为准。`completed`只表示执行完成，不表示候选性能或生产验收通过。

| 轮次 | 问题 / 候选 | 结论 |
|---|---|---|
| [122B原始研究](../docs/122b-original-report.md) | Hv16混合长短序列 | 混合CP有算子收益；未接入SGLang E2E |
| [20261023-shape-study](20261023-shape-study/README.md) | 35B TP1 CP扫描、V2、extend组件 | V2改善少序列；均衡合批需回退；当时无checkpoint支持 |
| [20261023-e2e-1p1d](20261023-e2e-1p1d/README.md) | 初次固定十条E2E | 旧hook覆盖有限；缓存条件不同于后续轮次 |
| [20261023-e2e-warm-1p1d](20261023-e2e-warm-1p1d/README.md) | 热前缀缓存条件 | 不能替代需要真实prefill的验证 |
| [20261024-e2e-sequential-1p1d](20261024-e2e-sequential-1p1d/README.md) | 清批次缓存、逐条运行 | 算子收益未直接转化为稳定TTFT，进入路径诊断 |
| [20261024-gdn-ttft-investigation](20261024-gdn-ttft-investigation/README.md) | 路径、BCG、checkpoint及等待诊断 | 确认需要覆盖实际checkpoint路径 |
| [20261024-gdn-checkpoint-opt](20261024-gdn-checkpoint-opt/README.md) | 单序列选定checkpoint | C1观测−12.64%；全路径auto-CP替换候选被拒绝 |
| [20261024-gdn-concurrency24](20261024-gdn-concurrency24/README.md) | 旧单序列patch在C2/C4 | 多序列回退多，收益不足且波动 |
| [20261024-gdn-multiseq-opt](20261024-gdn-multiseq-opt/README.md) | 多序列选定checkpoint + 自适应V | 当前核心版本；C2/C4平均改善，C8平均退化 |
| [20261024-gdn-chunk16k](20261024-gdn-chunk16k/README.md) | 两臂都用新GDN，8K→16Kchunk | 六个并发/轮次的平均TTFT均退化；拒绝16K默认值 |

MoE探索不属于本次GDN核心归档，未纳入。没有把不同实验混成一组结果，也没有把旧单序列实现包装成最终多序列版本。

## 后续统一与边界实验

- [跨geometry统一checkpoint](20261024-gdn-unified-checkpoint/README.md)：240项full/V2 exact核心检查、同版本16组对照；包含CP失败记录。
- [CP checkpoint边界修复](20261024-gdn-cp-checkpoint-fix/README.md)：native写出、112边界检查、200次动态回放、32项memcheck、16组性能；near-one失败仍保留。
- [统一公共入口验证](20261024-unified-entry/README.md)：新包prepare/run差分与graph/内存检查；不新增服务E2E。
