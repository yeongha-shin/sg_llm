import os
import cv2

def images_to_video(image_dir, output_path, fps=10):
    image_files = [
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png"))
    ]

    # 🔑 숫자 기준 정렬
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
    h, w, _ = first_img.shape

    video = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for fname in image_files:
        img = cv2.imread(os.path.join(image_dir, fname))
        video.write(img)

    video.release()
    print("Video saved:", output_path)


if __name__ == "__main__":
    images_to_video(
        image_dir="/home/jonghwi/pycharm/CoCaptain-Map/data/smd/1469",          # 이미지 폴더
        output_path="./output/video/smd_1469_test.mp4",     # 결과 비디오
        fps=10                          # 초당 프레임 수
    )
