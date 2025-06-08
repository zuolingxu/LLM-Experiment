from modelscope.utils.constant import Tasks
from modelscope import Model
from modelscope.pipelines import pipeline

model_name = "chatglm-6b"


model = Model.from_pretrained(model_name, device_map='auto', revision='v1.0.19').half().cuda()
pipe = pipeline(task=Tasks.chat, model=model)


while(True):
    usr_input = input("请输入问题:")
    if usr_input.lower() == "exit":
        print("Exiting...")
        exit()

    inputs = {'text': usr_input, 'history': []}
    result = pipe(inputs)

    for log in result['history']:
        print(log[1])