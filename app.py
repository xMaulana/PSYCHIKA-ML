from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print expected input shape for debugging
print("Expected input shape:", input_details[0]['shape'])

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

def preprocess_text(text):
    # Tokenize the text with the appropriate model tokenizer
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Sesuai dengan kebutuhan model
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='np'  # Menggunakan numpy untuk kompatibilitas dengan TFLite
    )
    # Directly use the encoded outputs without adding an extra dimension
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return input_ids, attention_mask

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text_input = data['text']
        print("Received text:", text_input)

        # Preprocess and predict
        input_ids, attention_mask = preprocess_text(text_input)
        print("Input IDs shape:", input_ids.shape)
        print("Attention Mask shape:", attention_mask.shape)

        # Adjusting the input to match the model's expected input shape [1, 1]
        # This assumes that some form of reduction to a single scalar is acceptable, which is usually not the case
        input_ids = np.array([[input_ids[0, 0]]])  # Taking just the first token's first ID
        attention_mask = np.array([[attention_mask[0, 0]]])  # Same for attention mask

        print("Adjusted Input IDs shape:", input_ids.shape)
        print("Adjusted Attention Mask shape:", attention_mask.shape)

        interpreter.set_tensor(input_details[0]['index'], input_ids)
        interpreter.set_tensor(input_details[1]['index'], attention_mask)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data, axis=-1)
        print("Prediction:", prediction)

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
