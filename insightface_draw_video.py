import numpy as np
import cv2
from insightface.app import FaceAnalysis


np.set_printoptions(suppress=True)
app = FaceAnalysis()  # 顔認識オブジェクト生成
app.prepare(ctx_id=-1, det_size=(640, 640))  # 前準備

# 点と口目に線を描画する関数
def draw(landmark, img):
    for i, point in enumerate(face.landmark_3d_68):
        x, y, z = point
        # 画像上の座標に変換
        x = int(x)
        y = int(y)
        # 円を描画
        img = cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        # 点同士を線でつなげる
        if i < 16:
            next_point = face.landmark_3d_68[i+1]
            next_x, next_y, next_z = next_point
            next_x = int(next_x)
            next_y = int(next_y)
            img = cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 2)
        elif 48 <= i <= 59:
            next_point2 = face.landmark_3d_68[i+1]
            next_x, next_y, next_z = next_point2
            next_x = int(next_x)
            next_y = int(next_y)
            img = cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 2)
        elif 36 <= i <= 40:
            next_point3 = face.landmark_3d_68[i+1]
            next_x, next_y, next_z = next_point3
            next_x = int(next_x)
            next_y = int(next_y)
            img = cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 2)
        elif 42 <= i <= 46:
            next_point = face.landmark_3d_68[i+1]
            next_x, next_y, next_z = next_point
            next_x = int(next_x)
            next_y = int(next_y)
            img = cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 2)
    return img

# ビデオを読み込み
cap = cv2.VideoCapture(0)

# csvに保存するためのリスト
yaw_list = []

while cap.isOpened():
    ret, img = cap.read()

    if ret:
        face = app.get(np.asarray(img))
        if len(face) == 0:
            continue
        face = face[0]
        
        # imgの高さと幅を取得
        h, w = img.shape[:2]
        _, yaw, _ = face.pose
        
        # yawをリストに追加
        yaw_list.append(yaw)
        
        img = draw(face.landmark_3d_68, img)
        cv2.putText(img, 'yaw: {:.0f}'.format(yaw), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
        
        cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# csvに保存
np.savetxt('yaw.csv', yaw_list, delimiter=',')

cap.release()
cv2.destroyAllWindows()