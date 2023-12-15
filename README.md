# YOLOv1_2020071685_Lee-Mijin
- 2023년 인공지능2 기말 과제_YOLOv1 리뷰_정보융합전공 2020071685 이미진

# YOLOv1
- 2016년 컴퓨터 비전 분야 최고 학회 중 하나인 CVPR에서 발표된 Object Detection(객체 인식) 모델
- 논문은 첨부파일의 'YOLOv1 논문.pdf'를 참고해주세요!
- 논문 원본 링크 https://arxiv.org/abs/1506.02640

  ### YOLOv1은 크게 3가지 특징을 가집니다.
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


