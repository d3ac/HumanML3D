# 你的四行文本标注数据
data_string = """
a man kicks something or someone with his left leg.#a/DET man/NOUN kick/VERB something/PRON or/CCONJ someone/PRON with/ADP his/DET left/ADJ leg/NOUN#0.0#0.0
the standing person kicks with their left foot before going back to their original stance.#the/DET stand/VERB person/NOUN kick/VERB with/ADP their/DET left/ADJ foot/NOUN before/ADP go/VERB back/ADV to/ADP their/DET original/ADJ stance/NOUN#0.0#0.0
a man kicks with something or someone with his left leg.#a/DET man/NOUN kick/VERB with/ADP something/PRON or/CCONJ someone/PRON with/ADP his/DET left/ADJ leg/NOUN#0.0#0.0
he is flying kick with his left leg#he/PRON is/AUX fly/VERB kick/NOUN with/ADP his/DET left/ADJ leg/NOUN#0.0#0.0
"""

# 用来存储解析后结果的列表
parsed_labels = []

# 按行读取数据
# .strip() 用于删除空行
for line in data_string.strip().split('\n'):
    # 使用 '#' 作为分隔符，将每一行拆分成四个部分
    parts = line.split('#')
    
    # 为了防止数据格式错误导致程序崩溃，检查一下是否成功拆分成了4部分
    if len(parts) == 4:
        # 将拆分后的结果存入一个字典中，这样更清晰
        label_info = {
            "raw_text": parts[0],
            "pos_tagged_text": parts[1],
            "value_1": float(parts[2]),  # 将字符串 '0.0' 转换为浮点数 0.0
            "value_2": float(parts[3])
        }
        parsed_labels.append(label_info)

# 打印结果，看看我们得到了什么
import json
print(json.dumps(parsed_labels, indent=2))