from __future__ import annotations

EVENT_TAXONOMY = [
    {
        "event": "在建农房",
        "domain": "治危拆违 / 农村建房监管",
        "definition": "农村居民点或农田周边出现正在建设、改扩建或未完工的农房建筑。",
        "positive": "村庄或农田旁有未完工建筑、裸露结构、施工材料、屋顶或墙体施工迹象。",
        "negative": "已完工普通民房；合法工地但非农房；普通庭院堆物。",
        "confusions": ["在建工程", "附属房建材", "房屋拆迁", "道路施工"],
        "evidence_types": ["Scene", "Object"],
    },
    {
        "event": "垃圾集中点",
        "domain": "市容环境 / 环境卫生",
        "definition": "居民区、道路边、空地或建筑周边出现集中堆放的生活垃圾、杂物或废弃物。",
        "positive": "成堆垃圾袋、杂物、废弃物集中在固定区域，规模明显。",
        "negative": "正常垃圾桶或垃圾站；少量零散杂物；建筑材料堆放。",
        "confusions": ["暴露垃圾", "建筑垃圾", "水边垃圾", "附属房建材"],
        "evidence_types": ["Region", "Object"],
    },
    {
        "event": "水面油污污染",
        "domain": "水域环保 / 水利",
        "definition": "水面出现油膜状、深色、彩虹反光或异常漂浮污染带，疑似油污污染。",
        "positive": "河道或池塘水面局部深色油膜、异常反光、边界较明显的污染斑块。",
        "negative": "正常水面阴影；云影或树影；黑臭水体整体发黑；水华浮膜。",
        "confusions": ["黑臭水体", "水华浮膜污染", "水面不洁", "水体异色污染"],
        "evidence_types": ["Region", "State"],
    },
    {
        "event": "流动摊贩",
        "domain": "城市管理 / 街面秩序",
        "definition": "道路、桥下、街边、空地等非固定经营区域出现流动售卖摊点或临时经营车辆/棚架。",
        "positive": "三轮车、货车、遮阳棚、摊位、人群和货品组合形成临时经营点。",
        "negative": "正常停放车辆；施工车辆；路边临时物料但无售卖行为。",
        "confusions": ["店外经营", "占道堆放", "消防通道占用", "车辆违停"],
        "evidence_types": ["Object", "Relation"],
    },
    {
        "event": "消防通道占用",
        "domain": "消防安全 / 城市管理",
        "definition": "消防通道、应急通道或小区消防车道被车辆、杂物、摊位、隔离物等占用。",
        "positive": "标识或道路形态显示为通道，通道上有车辆或物品阻挡通行。",
        "negative": "普通停车位车辆；非消防通道道路拥堵；短暂停靠但不阻断通行。",
        "confusions": ["车辆违停", "人行道车辆占用", "流动摊贩", "占道堆放"],
        "evidence_types": ["Relation", "Object"],
    },
    {
        "event": "烟雾排放",
        "domain": "生态环境 / 应急管理",
        "definition": "工厂、农田、工地、设施或地面源头持续排出明显烟雾、白烟、黑烟或灰烟。",
        "positive": "有明确烟柱或烟团从地面或设施向上扩散，源头可见或可推断。",
        "negative": "云雾、雾霾、尘土扬起、图像曝光造成的白斑。",
        "confusions": ["疑似烟火", "焚烧秸秆", "工地扬尘", "黑烟偷排"],
        "evidence_types": ["State", "Temporal"],
    },
    {
        "event": "疑似烟火",
        "domain": "消防安全 / 应急管理",
        "definition": "林地、村庄、农田或开放区域出现类似火点、烟柱、燃烧痕迹或火烟组合的疑似火情。",
        "positive": "小范围明火、烟火同现、林地或农田异常烟点、可能燃烧区域。",
        "negative": "工业白烟；水汽；尘土；烟雾排放但无火情风险。",
        "confusions": ["烟雾排放", "焚烧秸秆", "秸秆焚烧痕迹", "工地扬尘"],
        "evidence_types": ["State", "Temporal"],
    },
    {
        "event": "附属房建材",
        "domain": "住建 / 农村建房监管",
        "definition": "民房、附属房或院落周边堆放砖块、砂石、水泥、木板、钢材等建材，疑似施工或改建准备。",
        "positive": "房屋附近有明显建材堆、施工材料、临时堆放区。",
        "negative": "正常院落物品；厂区材料；建筑垃圾；道路施工材料。",
        "confusions": ["在建农房", "在建工程", "建筑垃圾", "厂区露天堆放", "房屋拆迁"],
        "evidence_types": ["Object", "Scene"],
    },
    {
        "event": "黑臭水体",
        "domain": "水域环保 / 城市治理",
        "definition": "河道、沟渠或池塘水体呈黑色、深褐色、浑浊或疑似严重污染状态。",
        "positive": "大面积水体发黑、发暗、与周边水体或正常水面差异明显。",
        "negative": "阴影、深水区、正常水塘颜色；局部油污；水华浮膜。",
        "confusions": ["水面油污污染", "水面不洁", "水华浮膜污染", "水体异色污染"],
        "evidence_types": ["Region", "State"],
    },
    {"event": "水边垃圾", "domain": "水域环保 / 环境卫生", "definition": "河岸、沟渠边、水塘边或水陆交界处堆积生活垃圾、杂物或废弃物。", "positive": "垃圾位于岸线附近、桥下、水边草丛或水陆交界处。", "negative": "水面漂浮垃圾；普通路边垃圾；建筑材料堆放。", "confusions": ["水面垃圾", "暴露垃圾", "垃圾集中点", "水面不洁"], "evidence_types": ["Region", "Relation"]},
    {"event": "水面不洁", "domain": "水域环保 / 水利", "definition": "水面存在明显浑浊、漂浮物、污染带、泡沫或异常反光，整体表现为水面环境不洁。", "positive": "水体表面有污浊斑块、泡沫、漂浮物或颜色异常。", "negative": "正常水面波纹；云影；水面油污污染的局部油膜；黑臭水体整体发黑。", "confusions": ["水面油污污染", "黑臭水体", "水面垃圾", "水华浮膜污染"], "evidence_types": ["Region", "State"]},
    {"event": "道路施工", "domain": "交通安全 / 住建", "definition": "道路、田间道路或城市道路正在建设、维修、开挖或铺设，存在施工区域。", "positive": "道路断面施工、土方、机械、围挡、临时施工带。", "negative": "普通道路；农田小路；在建农房周边地面裸露。", "confusions": ["在建工程", "桥梁施工", "道路破损", "附属房建材"], "evidence_types": ["Scene", "Object"]},
    {"event": "水华浮膜污染", "domain": "水域环保 / 水利", "definition": "水体表面出现绿色、黄绿色或片状浮膜，疑似藻类水华或漂浮生物污染。", "positive": "水面大面积绿色浮膜、藻华聚集、片状覆盖。", "negative": "油膜污染；黑臭水体；正常水草或岸边植物；水面垃圾。", "confusions": ["水面油污污染", "绿藻", "水面不洁", "黑臭水体"], "evidence_types": ["Region", "State"]},
    {"event": "秸秆焚烧痕迹", "domain": "农业农村 / 生态环境", "definition": "农田、地块或田埂附近出现秸秆焚烧后的黑色烧痕、灰烬、焦土或残留。", "positive": "田块上有明显黑色烧灼区域、条带状焦痕、灰烬残留。", "negative": "普通裸土；阴影；收割后的田地纹理；道路施工痕迹。", "confusions": ["焚烧秸秆", "疑似烟火", "烟雾排放", "裸土未覆盖"], "evidence_types": ["Region", "State"]},
    {"event": "暴露垃圾", "domain": "市容环境 / 环境卫生", "definition": "公共区域、村庄、道路边或空地中出现未入容器、裸露堆放的生活垃圾。", "positive": "成片垃圾、塑料袋、废弃物裸露在地面。", "negative": "建筑垃圾；可用物料堆放；水边垃圾；正常垃圾桶内垃圾。", "confusions": ["垃圾集中点", "水边垃圾", "建筑垃圾", "乱堆物料"], "evidence_types": ["Region", "Object"]},
    {"event": "水面垃圾", "domain": "水域环保 / 城市管理", "definition": "河道、池塘、沟渠等水面漂浮塑料、泡沫、枝叶、袋子等垃圾。", "positive": "垃圾位于水面上，随水漂浮或聚集。", "negative": "岸边垃圾；水华浮膜；水面反光；油膜污染。", "confusions": ["水边垃圾", "水面不洁", "水华浮膜污染", "水面油污污染"], "evidence_types": ["Region", "Object"]},
    {"event": "厂区露天堆放", "domain": "安全生产 / 城市管理", "definition": "厂区、仓储区或工业场地内露天堆放材料、设备、废料或杂物。", "positive": "工业厂房周边露天堆积大量物料、桶、设备、废弃物。", "negative": "农房建材；施工现场建材；建筑垃圾。", "confusions": ["附属房建材", "建筑垃圾", "厂区堆积物", "在建工程"], "evidence_types": ["Scene", "Object"]},
    {"event": "在建工程", "domain": "住建 / 安全生产", "definition": "城市、乡镇或工地范围内有正在施工建设的工程项目。", "positive": "脚手架、裸露地基、施工区域、工程材料、施工机械、未完工建筑。", "negative": "在建农房；房屋拆迁；道路施工；普通空地。", "confusions": ["在建农房", "道路施工", "附属房建材", "房屋拆迁"], "evidence_types": ["Scene", "Object"]},
    {"event": "建筑垃圾", "domain": "市容环境 / 住建", "definition": "建筑施工、拆除、装修后产生的砖块、混凝土、砂石、木板等废弃物堆放。", "positive": "废砖、混凝土块、施工废料、拆除残渣集中堆积。", "negative": "可用建材；生活垃圾；厂区物料；农田裸土。", "confusions": ["附属房建材", "垃圾集中点", "暴露垃圾", "房屋拆迁"], "evidence_types": ["Region", "Object"]},
    {"event": "房屋拆迁", "domain": "住建 / 治危拆违", "definition": "建筑物正在拆除或拆除后形成残垣、裸露结构、拆迁废墟。", "positive": "房屋局部拆毁、屋顶或墙体破损、拆除机械或大面积废墟。", "negative": "在建工程；危房隐患；建筑垃圾单独堆放；农房建设。", "confusions": ["建筑垃圾", "在建工程", "危房隐患", "附属房建材"], "evidence_types": ["Scene", "State"]},
]

PRIMARY_EVENTS = {
    "在建农房",
    "垃圾集中点",
    "水面油污污染",
    "流动摊贩",
    "消防通道占用",
    "烟雾排放",
    "疑似烟火",
    "附属房建材",
    "黑臭水体",
}


def event_lookup() -> dict[str, dict]:
    return {row["event"]: row for row in EVENT_TAXONOMY}


def normalize_event_name(name: str) -> str:
    value = name.strip()
    for suffix in ("坐标图", "图片", "示例"):
        value = value.replace(suffix, "")
    return value.strip(" _-，,、.。")

