# #torch,simletrans,flask 
# from flask import Flask,request,jsonify
# from simpletransformers.classification import ClassificationModel, ClassificationArgs

# app = Flask(__name__)
# model_path = "interface\model1"

# @app.route('/',methods=['POST'])
# def model():
#     data = request.json
#     print(data)
#     model = ClassificationModel(model_type="bert",model_name= model_path,use_cuda=False)
#     pred,_ = model.predict(data["text"])
#     res = {
#         "data":str(pred)
#     }
#     return jsonify(res)


# if __name__ == '__main__':
#     app.run()

from flask import Flask, request, jsonify
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np

app = Flask(__name__)

modelp_path = "interface/modelprof" 
models_path = "interface/modelsal" 

model1 = ClassificationModel(model_type="bert", model_name=modelp_path, use_cuda=False)
model2 = ClassificationModel(model_type="bert", model_name=models_path, use_cuda=False)

@app.route('/', methods=['POST'])
def classify_text():
    data = request.json

    # 'text' is the key for the list of text data in JSON request
    texts_to_classify = data.get("text", [])

    result1, _ = model1.predict(texts_to_classify)

    non_profane_indices = np.where(result1 == 1)[0]
    non_profane_sentences = [texts_to_classify[i] for i in non_profane_indices]

    result2, _ = model2.predict(non_profane_sentences)
    response = {
        "Salient Sentences": [non_profane_sentences[i] for i in range(len(result2)) if result2[i] == 1],
        "Non-salient Sentences which will be removed": [non_profane_sentences[i] for i in range(len(result2)) if result2[i] == 0],
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
