import tensorflow as tf
import argparse

def convert_model(keras_model_path, tflite_model_path):
    # Загрузка Keras модели
    model = tf.keras.models.load_model(keras_model_path)
    
    # Конвертация в TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Сохранение модели
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model", required=True)
    parser.add_argument("--tflite_model", required=True)
    args = parser.parse_args()
    
    convert_model(args.keras_model, args.tflite_model)