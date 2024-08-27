import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir="./models")

# 模型初始化
path = './models/IEITYuan/Yuan2-2B-Mars-hf'
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-650'
torch_dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 自定义出题函数
def chuti(text, num_questions, num_choices):
    template = f'''
    你是一名经验丰富的高考语文和英语命题专家。

    所给的阅读文本如下：
    {text}

    请你出{num_questions}道不同的单项选择题目，每个题目有{num_choices}个选项；
    然后对每个题对应给出答案，解析，出题思路/考察能力。

    出题示例如下：

    1.题目：xxx
    A.xxx
    B.xxx
    C.xxx
    D.xxx
    答案和解析：xxx

    (注意换行)
    '''

    prompt = template.strip()
    prompt += "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(inputs, do_sample=False, max_length=8000)
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '').strip()
    response = response.replace("\n", "  \n")
    return response


# Sample text examples
example_1 = "社 戏 （节选）\n沈从文\n萝卜溪邀约的浦市戏班子，赶到了吕家坪，是九月二十二。一行十四个人，八个笨大衣箱，坐了只辰溪县装石灰的空船，到地时，便把船靠泊在码头边。掌班依照老规矩，带了个八寸大的朱红拜帖，来拜会本村首事滕长顺，商量看是在什么地方搭台，哪一天起始开锣。\n半月来省里向上调兵开拔的事，已传遍了吕家坪。不过商会会长却拿定了主意：照原来计划装了五船货物向下游放去。长顺因为儿子三黑子的船已到地卸货，听会长亲家出主意，也预备装一船橘子下常德府。空船停泊在河边，随时有人把黄澄澄的橘子挑上船，倒进舱里去。戏班子乘坐那只大空船，就停靠在橘子园边不多远。\n两个做丑角的浦市人，扳着船篷和三黑子说笑话，以为古来仙人坐在斗大橘子中下棋，如今仙人坐在碗口大橘子堆上吸烟，世界既变了，什么都得变。可是三黑子却想起保安队队长向家中讹诈事情，因此只向那个做丑角的戏子苦笑。\n长顺约集本村人在伏波宫开会，商量看这戏演不演出。时局既不大好，集众唱戏是不是影响治安？末了依照多数主张，班子既然接来了，酬神戏还是在伏波宫前空坪中举行。凡事依照往年成例，出公份子演戏六天，定二十五开锣。并由本村出名，具全红帖子请了吕家坪的商会会长，和其他庄口上的有名人物，并保安队队长、排长、师爷、税局主任、督察等，到时前来看戏。还每天特别备办两桌四盘四碗酒席，款待这些人物。\n到开锣那天，本村和附近村子里的人，都换了浆洗过的新衣服，荷包中装满零用钱，赶到萝卜溪伏波宫看大戏。因为一有戏，照习惯吕家坪镇上卖大面的、卖豆糕米粉的、油炸饼和其他干湿甜酸熟食冷食的，无不挑了锅罐来搭棚子，竞争招揽买卖。妇女们且多戴上满头新洗过的首饰，或镀金首饰，发蓝点翠首饰，扛一条高脚长板凳，成群结伴跑来看戏，必到把入晚最后一幕杂戏看完，把荷包中零用钱花完，方又扛起那条凳子回家。有的来时还带了饭箩和针线，有的又带了香烛纸张顺便敬神还愿。平时单纯沉静的萝卜溪，于是忽然显得空前活泼热闹起来。\n长顺一家正忙着把橘子下树上船，还要为款待远来看戏亲友，准备茶饭，因此更见得热闹而忙乱。家中每天必为镇上和其他村子里来的客人，办一顿过午面饭。又另外烧了几缸热茶，供给普通乡下人。长顺自己且换了件大船主穿的大袖短摆蓝宁绸长衫，罩一件玄青羽绫马褂，舞着那个挂有镶银老虎爪的紫竹马鞭长烟杆，到处走动拜客。\n第一天开锣时，由长顺和其他三个上年纪的首事人，在伏波爷爷神像前磕头焚香，杀了一只白羊，一只雄鸡，烧了个申神黄表。戏还未开场，空坪中即已填满了观众，吕家坪的官商要人，都已就座。开锣后即照例“打加官”，由一个套白面具的判官，舞着个肮脏的红缎巾幅，台上打小锣的检场人叫一声：“某大老爷禄位高升！”那判官即将巾幅展开，露出字面。被尊敬颂祝的，即照例赏个红包封。有的把包封派人送去，有的表示豪爽，便把那个赏金用力直向台上掼去，惹得一片喝彩。当天第一个叫保安队队长。第一出戏象征吉祥性质，对神示敬，对人颂祷。第二出戏与劝忠教孝有关。到中午休息，匀出时间大吃大喝。休息时间，一些戏子头上都罩着发网子，脸上颜料油腻也未去净，争到台边熟食棚子去喝酒，引得观众包围了棚子看热闹。妇女们把扣双凤桃梅大花鞋的两脚，搁在高台子踏板上，口中嘘嘘的吃辣子羊肉面，或一面剥葵花子，一面并谈论做梦绩麻琐碎事情。下午开锣重唱，戏文转趋热闹活泼。\n掌班走到几位要人身边来请求赏脸，在排定戏目外额外点戏。\n大家都客气谦让，不肯开口。经过一阵撺掇，队长和税局主任是远客，少不了各点一出，会长也被迫点一出。队长点“武松打虎”，因为武人点英雄，短而热闹，且合身份；会长却点“王大娘补缸”，戏是趣剧，用意在于与民同乐。戏文经点定后，照例也在台柱边水牌上写明白，给看戏人知道。开锣后正角上场，又是包封赏号，这个包封却照例早由萝卜溪办会的预备好，不用贵客另外破钞。\n最末一出杂戏多是短打，三个穿红裤子的小花脸，在台上不住翻跟斗，说浑话。\n收锣时已天近黄昏，天上一片霞，照得人特别好看。一切人影子都被斜阳拉得长长的，脸庞被夕阳照炙得红红的。到处是笑语嘈杂，过吕家坪去的渡头，尤其热闹。方头平底大渡船，装满了从戏场回家的人，慢慢在平静河水中移动，两岸小山都成一片紫色，天上云影也逐渐在由黄而变红，由红而变紫。太空无云处但见一片深青，秋天来特有的澄清。在淡青色天末，一颗长庚星白金似的放着煜煜光亮，慢慢地向上升起。远山野烧，因逼近薄暮，背景既转成深蓝色，已由一片白烟变成点点红火。……一切光景无不神奇而动人。可是人人都融和在这种光景中，带点快乐和疲倦的心情，等待还家。无一个人能远离这个社会的快乐和疲倦，声音与颜色，来领会赞赏这耳目官觉所感受的新奇。\n（有删改）"
example_2 = "In the race to document the species on Earth before they go extinct, researchers and citizen scientists have collected billions of records. Today, most records of biodiversity are often in the form of photos, videos, and other digital records. Though they are useful for detecting shifts in the number and variety of species in an area, a new Stanford study has found that this type of record is not perfect.\n“With the rise of technology it is easy for people to make observations of different species with the aid of a mobile application,” said Barnabas Daru, who is lead author of the study and assistant professor of biology in the Stanford School of Humanities and Sciences. “These observations now outnumber the primary data that comes from physical specimens (标本), and since we are increasingly using observational data to investigate how species are responding to global change, I wanted to know: Are they usable?”\nUsing a global dataset of 1.9 billion records of plants, insects, birds, and animals, Daru and his team tested how well these data represent actual global biodiversity patterns.\n“We were particularly interested in exploring the aspects of sampling that tend to bias (使有偏差) data, like the greater likelihood of a citizen scientist to take a picture of a flowering plant instead of the grass right next to it,” said Daru.\nTheir study revealed that the large number of observation-only records did not lead to better global coverage. Moreover, these data are biased and favor certain regions, time periods, and species. This makes sense because the people who get observational biodiversity data on mobile devices are often citizen scientists recording their encounters with species in areas nearby. These data are also biased toward certain species with attractive or eye-catching features.\nWhat can we do with the imperfect datasets of biodiversity?\n“Quite a lot,” Daru explained. “Biodiversity apps can use our study results to inform users of oversampled areas and lead them to places – and even species – that are not well-sampled. To improve the quality of observational data, biodiversity apps can also encourage users to have an expert confirm the identification of their uploaded image.”"

# Streamlit UI
st.title("AI 高考出题助手")

st.write("输入文本或选择范例后选择题目数量和选项数量，然后点击按钮生成题目及答案解析。")

# Toggle between input or example
input_mode = st.radio("选择输入方式", options=["自定义输入", "选择范例"], horizontal=True)

if input_mode == "自定义输入":
    input_text = st.text_area("输入阅读材料", height=300)
else:
    example_selected = st.radio("选择范例", options=["中文范例", "英文范例"], horizontal=True)
    if example_selected == "中文范例":
        input_text = example_1
    else:
        input_text = example_2
    st.write("**范例内容:**")
    st.write(input_text)

question_count = st.radio("选择生成题目数量", options=[1, 2, 3, 4, 5], index=1, horizontal=True)
option_count = st.radio("选择选项数量", options=[3, 4], index=1, horizontal=True)

if st.button("生成题目"):
    if input_text.strip() != "":
        with st.spinner("正在生成中，请稍候..."):
            result = chuti(input_text, question_count, option_count)
        
        # Display the input text and generated questions side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("输入材料")
            st.write(input_text)
        
        with col2:
            st.header("生成的题目")
            st.write(result, unsafe_allow_html=True)
    else:
        st.warning("请先输入文本材料或选择范例。")
