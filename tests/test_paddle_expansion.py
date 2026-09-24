from __future__ import annotations

from modelome.sources.paddle_model_center import _parse_info


def test_model_center_retains_chinese_only_description_and_task_metadata() -> None:
    info = _parse_info(
        "PP-Example",
        "modelcenter/PP-Example/info.yaml",
        """\
Model_Info:
  name: PP-Example
  description_cn: 面向图像分类的轻量级模型
Task:
  - tag_cn: 计算机视觉
    sub_tag_cn: 图像分类
""",
        "paddle-model-center",
    )

    assert info.description == "面向图像分类的轻量级模型"
    assert info.tasks == ("计算机视觉", "图像分类")


def test_model_center_prefers_english_metadata_when_both_languages_are_declared() -> None:
    info = _parse_info(
        "PP-Example",
        "modelcenter/PP-Example/info.yaml",
        """\
Model_Info:
  name: PP-Example
  description_en: Lightweight image classifier
  description_cn: 面向图像分类的轻量级模型
Task:
  - tag_en: Computer Vision
    tag_cn: 计算机视觉
    sub_tag_en: Image Classification
    sub_tag_cn: 图像分类
""",
        "paddle-model-center",
    )

    assert info.description == "Lightweight image classifier"
    assert info.tasks == (
        "Computer Vision",
        "计算机视觉",
        "Image Classification",
        "图像分类",
    )
