import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def prepare_gtsrb_dataset():
    # 1. Создаем основной DataFrame для GTSRB
    gtsrb_data = []
    for class_id in range(0, 43):
        class_dir = f"dataset/GTSRB/Final_Training/Images/{class_id:05d}"
        annotations_file = f"{class_dir}/GT-{class_id:05d}.csv"
        
        if os.path.exists(annotations_file):
            class_df = pd.read_csv(annotations_file, sep=';')
            class_df['ClassId'] = class_id
            gtsrb_data.append(class_df)
    
    gtsrb_df = pd.concat(gtsrb_data)
    gtsrb_df['Path'] = gtsrb_df['Filename'].apply(
        lambda x: f"dataset/GTSRB/Final_Training/Images/{gtsrb_df['ClassId']:05d}/{x}"
    )
    
    return gtsrb_df[['Path', 'ClassId']]

def prepare_new_data():
    # 2. Создаем DataFrame для новых данных
    new_data = []
    for img_file in os.listdir("data"):
        if img_file.endswith((".png", ".jpg")):
            label_file = os.path.join("labels", os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    label = int(f.read().strip())
                new_data.append({
                    "Filename": img_file,
                    "ClassId": label,
                    "Path": os.path.join("data", img_file)
                })
    
    return pd.DataFrame(new_data)

def create_final_dataset():
    # 3. Объединяем данные
    gtsrb_df = prepare_gtsrb_dataset()
    new_df = prepare_new_data()
    
    full_df = pd.concat([gtsrb_df, new_df[['Path', 'ClassId']]])
    
    # 4. Сохраняем финальный CSV
    full_df.to_csv("dataset/annotations.csv", index=False)
    
    # 5. Копируем новые изображения в общую папку
    os.makedirs("dataset/images", exist_ok=True)
    for _, row in new_df.iterrows():
        shutil.copy(row["Path"], "dataset/images/")

if __name__ == "__main__":
    create_final_dataset()