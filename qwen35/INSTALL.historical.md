# 35B TP1 H20 selected checkpoint GDN

此目录7个Python文件需整体使用，不能混用旧单序列patch。仅针对已验证的H20 78SM / SGLang0.5.20 / FlashInfer0.6.18与35B TP1形状。仅P进程设置：

```sh
PYTHONPATH=/absolute/path/to/release GDN35_PATCH=1 GDN35_VARIANT=multi python3 -m dynamo.sglang [现有P参数]
```

固定multi模式不读取外部实验目录的mode文件。D不设置上述hook环境。撤销时移除这两个环境变量并重启P。不修改安装包。

Checkpoint planner、packed metadata消费者和两个kernel必须整体加载，私有负interval不是上游公开协议。无checkpoint的auto-CP保持原版。N>8或不支持形状回退原版。

900次真实调用一致，72例合成、277映射、8memcheck、6真实fixture验证通过。详情见实验README；这不是长期soak/模型质量认证。C8端到端ABBA结果仍退化，不能承诺生产收益。
