# YOLOv1_2020071685_Lee-Mijin
- 2023년 인공지능2 기말 과제_YOLOv1 리뷰_정보융합전공 2020071685 이미진
- 본 게시물은 2023년 한양대학교 정보융합전공 '인공지능2' 과목 기말 과제입니다.
- YOLOv1에 대해 보다 쉽게 이해하실 수 있도록 논문을 기반으로 설명하며, 관련 코드 구현 페이지를 소개합니다.
- 관련 발표 자료는 첨부 파일의 'YOLOv1 논문 발표 자료'에서 확인할 수 있습니다.

# ⭐YOLOv1⭐
- 2016년 컴퓨터 비전 분야 최고 학회 중 하나인 CVPR에서 발표된 Object Detection(객체 인식) 모델
- 논문은 첨부파일의 'YOLOv1 논문.pdf'를 참고해주세요!
- 논문 원본 링크 https://arxiv.org/abs/1506.02640


**YOLOv1은 크게 3가지 특징을 가집니다.**
1) You Only Look Once
- 전체 이미지를 모델에 한번 통과
- 즉, 이미지 전체를 딱 한번만 봄
2) Unified
- 통합된 모델
- Region Proposal과 Feature Extraction 단계를 통합한 1 stage 방법
- 쉽게 말해 '물체가 어디에 위치해 있는지?'와 '그 물체가 무엇인지?'를 한번에 수행
3) Real Time
- 1 stage 방법으로 동작 속도가 매우 빨라짐
- 이에 Real Time Object Detection(실시간 객체 탐지) 가능
- 관련 영상 https://www.youtube.com/watch?v=K9a6mGNmhbc

## 0️⃣YOLOv1 제안방법
![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/73bda07b-b6dd-4997-8390-940f05149fdc)
(이미지 출처: 논문 원본)
1. 448*448 사이즈로 이미지 resize
2. 단 1개의 CNN 네트워크에 통과
3. NMS(Non-Maximum Suppression)을 통해 최종 Bounding Box의 location과 class 결정
    ==> 즉, 어디에 객체가 있는지, 이 객체가 무엇인지 결정!

크게는 위와 같은 3단계를 통해 YOLOv1이 수행됩니다.

## 1️⃣YOLOv1 CNN 제안방법
앞선 'YOLOv1 제안방법'의 2에 해당하는 CNN 시스템에 대해 말씀드리겠습니다.
![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/6e0878a1-2dd6-41b5-a7ae-75c1327f732e)
