<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/>
# 소개
이 저장소는 OpenCV를 활용한 몇가지 간단한 프로젝트를 담고 있습니다.

## A2_2d_transformation.py

몇가지 명령어를 입력받아 이미지를 이동, 회전, 확대 및 축소 작업을 수행하는 코드를 작성하였습니다.
이미지의 변환은 각 픽셀값에 변환행렬을 곱하는 것으로 구현되었습니다.
가능한 명령어는 아래와 같습니다.
‘a’ Move to the left by 5 pixels
‘d’ Move to the right by 5 pixels
‘w’ Move to the upward by 5 pixels
‘s’ Move to the downward by 5 pixels
‘r’ Rotate counter-clockwise by 5 degrees
‘R’ Rotate clockwise by 5 degrees
‘f’ Flip across 𝑦𝑦 axis
‘F’ Flip across 𝑥𝑥 axis
‘x’ Shirnk the size by 5% along to 𝑥𝑥 direction
‘X’ Enlarge the size by 5% along to 𝑥𝑥 direction
‘y’ Shirnk the size by 5% along to 𝑦𝑦 direction
‘Y’ Enlarge the size by 5% along to 𝑦𝑦 direction
‘H’ Restore to the initial state
‘Q’ Quit

## A2_homography.py

두 이미지로부터 feature를 추출하여 각 feature들을 매칭한 뒤 homography matrix를 구할 수 있습니다.
Feature들을 매칭할 때 단순히 feature간의 해밍 거리를 이용하는 방법(compute_homography 함수)과 RANSAC 알고리즘을 이용하는 방법(compute_homography_ransac) 두가지가 있습니다.
이렇게 구한 homography matrix를 활용해 특정 이미지에 다른 이미지를 덮어씌우거나 이미지끼리 이어붙이는 등의 작업을 수행할 수 있습니다.
