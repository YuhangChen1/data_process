"""Five difficulties within temporal Level 1, distinct from Level 2 changes."""
from string import Formatter

VERSION = 'frame_qa/1.0'
LEVELS = {
    1: ('recognition', '可见物体识别',
        '请根据当前帧的已标注可见物体，完整回答看到了哪些独立物体和连续表面。'
        '区分物体类别与表面，不把分割片段数当成真实物体总数。'),
    2: ('attributes', '物体与外观几何特征',
        '先识别物体，再说明主要颜色、可见轮廓和几何特征。颜色取自当前帧对应 mask，'
        '几何只描述已观测部分，不能把表面范围当作完整实物尺寸或凭颜色猜材质。'),
    3: ('relative_space', '物体间相对空间',
        '先描述物体特征，再说明它与其他可见物体的相对位置。左右和上下必须说明是'
        '当前图像坐标关系；canonical 轴方向和三维距离依据已提供真值，不能猜接触或支撑关系。'),
    4: ('global_space', '世界坐标与绝对位置',
        '识别物体、描述特征，并报告 canonical 世界坐标系下观测表面包围框的中心坐标、'
        '三轴范围及所在楼层。Z 朝上，单位 inferred_meter，不虚构东南西北、房间名或完整尺寸。'),
    5: ('camera_space', '相对拍摄相机的远近',
        '先识别用于比较的物体及特征，再依据当前帧相机位姿和深度一致的可见表面点，'
        '比较谁离相机近、谁远。使用相机到可见表面点的三维欧氏距离中位数，不能用屏幕大小'
        '或世界坐标 Z 替代距离；差异不足或缺少有效点时明确无法可靠排序。'),
}

# GPT writes the language once from real supplied annotations. Numeric facts and
# IDs are filled by code; this is deliberately recorded as template generation,
# never represented as thousands of independent GPT image inferences.
SLOTS = {
    1: {'inventory', 'surfaces', 'inventory_note'},
    2: {'focus_names', 'attribute_details'},
    3: {'focus_names', 'attribute_details', 'relative_details'},
    4: {'focus_names', 'attribute_details', 'global_details', 'coordinate_convention'},
    5: {'pair_names', 'pair_attributes', 'camera_details'},
}
REQUIRED = {
    1: {'inventory', 'surfaces', 'inventory_note'},
    2: {'attribute_details'},
    3: {'attribute_details', 'relative_details'},
    4: {'attribute_details', 'global_details', 'coordinate_convention'},
    5: {'pair_attributes', 'camera_details'},
}


def bank_prompt(examples):
    import json
    return '''为住宅重建数据的“Level 1 帧内问答”设计五档稳定中文问题及回答句式。
Level 1 是时间范围，不是下方的难度。五档难度按如下要求递进：
''' + json.dumps({str(k): {'name': v[1], 'prompt': v[2]} for k, v in LEVELS.items()}, ensure_ascii=False) + '''
直接给你已有物体、实例 ID、空间数据及由此构造的回答槽位示例。请以提供的标注为回答依据，
无需重新识别图片，也不要生成新的物体、属性或数字。示例中的自动标注仍未人工审核。
你的任务是生成问句和可重复填充的自然语言回答句式，代码会把不同帧的事实填入花括号槽位。
每档只生成一对稳定句式，不输出具体示例答案。语气自然、具体、完整，避免“这是啥”“是什么颜色”
等过短问题，也避免泛泛填充和把要求变成只有“是/否”的问题。
第1档要求盘点当前可见的已识别类别和连续表面；第2档要求识别并描述颜色、轮廓；
第3档增加物体间有参考系的空间关系；第4档需要世界坐标和观测范围；第5档先识别再比较相机远近。
问句中不要出现实例编号、文件路径、答案内容、具体坐标/距离、或从样本记住的固定物体名称。
第2至4档问句可用 {focus_names} 指定可见对象，第5档可用 {pair_names}，但不硬编码名称。
第1档无需槽位。各档答案必须有其 required_answer_slots 中的全部槽位，可有简短自然衔接。
不要在句式中额外陈述颜色、空间关系、人数、材质、朝向或测量值；这些必须来自事实槽位。
仅允许单层花括号变量，无 Python 表达式、格式符、属性访问或其他模板语法。
不要求或推断真实物体运动。不要把大类/表面重复片段当成精确实物数量。
第4档测量需要组织方三维真值；不要宣称只看 RGB 就能知道绝对坐标。
第5档距离比较是可见表面统计量，不能冒充完整物体中心距离或可导航距离。
返回 JSON：{"levels":[{"difficulty_level":1,"question_template":"当前画面中有哪些可辨认的物体和连续表面？请分别说明。","answer_template":"可辨认的物体包括{inventory}。可见连续表面包括{surfaces}。{inventory_note}"}, ...五项]}
允许槽位及每档必需的回答槽位：
''' + json.dumps({str(k): {'allowed_slots': sorted(SLOTS[k]), 'required_answer_slots': sorted(REQUIRED[k])} for k in LEVELS}, ensure_ascii=False) + '''
下面是来自真实 household 帧的结构化真值样本（仅作为句式设计依据，不要将样本常量写入模板）：
''' + json.dumps(examples, ensure_ascii=False, separators=(',', ':'))


def template_fields(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Empty GPT template')
    fields = set()
    for _, field, spec, conversion in Formatter().parse(value):
        if field is None:
            continue
        if spec or conversion or not field.isidentifier():
            raise ValueError('Unsafe template syntax')
        fields.add(field)
    return fields


def validate_bank(parsed):
    rows = parsed.get('levels', [])
    if len(rows) != 5:
        raise ValueError('GPT must return all five difficulty levels')
    found = {}
    for row in rows:
        level = row.get('difficulty_level')
        if type(level) is not int or level not in LEVELS or level in found:
            raise ValueError('Invalid/duplicate difficulty level')
        qfields = template_fields(row.get('question_template'))
        afields = template_fields(row.get('answer_template'))
        if not qfields <= ({'focus_names'} if level in (2, 3, 4) else {'pair_names'} if level == 5 else set()):
            raise ValueError('Question leaks answer slots')
        if not REQUIRED[level] <= afields or not afields <= SLOTS[level]:
            raise ValueError('Missing/unsupported answer slots')
        if level in (2, 3, 4) and 'focus_names' not in qfields:
            raise ValueError('Attribute/spatial question must bind the focus objects')
        if level == 5 and 'pair_names' not in qfields:
            raise ValueError('Camera question must bind the compared objects')
        found[level] = row
    return found
