from collections import Counter
from unstructured.partition.pdf import partition_pdf

# PDF 文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"


def summarize_elements(elements, label):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

    types = Counter(e.category for e in elements)
    print(f"元素类型: {dict(types)}")

    print("\n前 10 个元素:")
    for i, element in enumerate(elements[:10], 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("-" * 60)


# 1) hi_res 策略
hi_res_elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
)

# 2) ocr_only 策略
ocr_only_elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
)

# 分别输出结果
summarize_elements(hi_res_elements, "hi_res")
summarize_elements(ocr_only_elements, "ocr_only")

# 简单对比
print(f"\n{'=' * 20} 对比总结 {'=' * 20}")
print(f"hi_res 元素数: {len(hi_res_elements)}")
print(f"ocr_only 元素数: {len(ocr_only_elements)}")
print(
    "通常 hi_res 更擅长保留版面结构（如标题、段落、表格边界等），"
    "而 ocr_only 更偏向直接从页面做 OCR 提取纯文本。"
)