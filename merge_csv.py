import pandas as pd
import os
import tqdm

# Thư mục chứa các file CSV
folder_path = 'data'

# Lấy danh sách tất cả các file CSV trong thư mục
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Khởi tạo một danh sách để chứa các DataFrame
dataframes = []

# Đọc từng file CSV và thêm vào danh sách
for file in tqdm.tqdm(csv_files):
    df = pd.read_csv(os.path.join(folder_path, file))
    dataframes.append(df)

# Gộp tất cả các DataFrame lại thành một DataFrame duy nhất
total_df = pd.concat(dataframes, ignore_index=True)

# Xuất DataFrame gộp ra file CSV mới
total_df.to_csv('total.csv', index=False)

print("Gộp CSV thành công!")
