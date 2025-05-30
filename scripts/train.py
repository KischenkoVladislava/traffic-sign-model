import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(path, target_size=(32, 32)):
    img = load_img(path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array

def build_model(num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # 1. Загрузка аннотаций
    df = pd.read_csv("dataset/annotations.csv")
    
    # 2. Подготовка данных
    print("Loading images...")
    X = np.array([load_and_preprocess_image(path) for path in df['Path']])
    y = tf.keras.utils.to_categorical(df['ClassId'], num_classes=43)
    
    # 3. Разделение данных
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Создание и обучение модели
    model = build_model(43)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val)
    )
    
    # 5. Сохранение модели
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()