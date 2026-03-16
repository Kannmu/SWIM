# CLAUDE.md


- 项目的 Conda Python 虚拟环境名称为：`DLM`
- 项目的 Python 版本为 `3.12`
- 项目所使用的 Python GUI库为：`CustomTkinter`
- Manuscript 和 Supplementary Information 中严格报错引用的.bib 文件为：`SWIM.bib`。其他.bib文件都仅用于信息搜索和获取论文内容总结，不参与编译。在其他.bib文件中发现的论文的 citation key 都需要在`SWIM.bib`中确认存在之后，再进行引用。所有citation key都只能从`SWIM.bib`中获取。
- 引用使用`\cref{}`命令，例如：`\cref{fig:umh_device}`。
- 图片中的文字使用`Sentence Case`的大小写规范，每条只有第一个单次的首字母大写，其他单次字母小写。
- 每张图片，每个表格都要在文中有对应的引用。

- 不同调制模型可视化配色：

```py
METHOD_COLORS = {
    'ULM_L': '#443983',
    'DLM_2': '#31688E',
    'DLM_3': '#21918C',
    'LM_C': '#35B779',
    'LM_L': '#90D743',
}
```

