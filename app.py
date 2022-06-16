from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from werkzeug.exceptions import BadRequest
from flask import Flask, request, make_response
import numpy as np

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/', methods=['POST'])
def get_bot_message():
    if 'message' not in request.json:
        raise BadRequest("Missing message parameter!")
    message = request.json['message']
    if message == '':
        raise BadRequest("Invalid message!")
    new_user_input_ids = tokenizer.encode(
        message + tokenizer.eos_token, return_tensors='pt')
    with open('./chat_history_ids.txt') as f:
        chat_history_ids = f.read().split(' ')
        chat_history_ids = None if chat_history_ids == [''] else [int(id) for id in chat_history_ids]
    if chat_history_ids is not None:
        chat_history_ids = np.array([chat_history_ids])
        chat_history_ids = torch.from_numpy(chat_history_ids)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    output_message = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    with open('./chat_history_ids.txt', 'w') as f:
        save_chat_history_ids = chat_history_ids.numpy()
        save_chat_history_ids = save_chat_history_ids.tolist()
        write_string = " ".join(str(id) for id in save_chat_history_ids[0])
        f.write(write_string)
    response = make_response(output_message)
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
