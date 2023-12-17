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


   **1. 448*448 사이즈로 이미지 resize**

   **2. 단 1개의 CNN 네트워크에 통과**

   **3. NMS(Non-Maximum Suppression)을 통해 최종 Bounding Box의 location과 class 결정**

==> 즉, 어디에 객체가 있는지, 이 객체가 무엇인지 결정!


크게는 위와 같은 3단계를 통해 YOLOv1이 수행됩니다.

## 1️⃣YOLOv1 CNN 제안방법
앞선 'YOLOv1 제안방법'의 2에 해당하는 CNN 시스템은 다음과 같습니다.
![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/6e0878a1-2dd6-41b5-a7ae-75c1327f732e)
(이미지 출처: 논문 원본)


*Confidence: 해당 박스 안에 물체가 있을 확률*

*C개의 조건부 확률: 해당 박스안에 물체가 있을 때, t번째 클래스일 확률*


   **1. 이미지를 S*S 그리드 셀로 나눈다.**

   **2. 각각의 그리드에서 물체가 있을 만한 영역을 B개의 Bounding Box로 예측한다.**
   
   **3. 박스의 Confidence를 예측한다.**
  
   **4. 각각의 그리드마다 C개의 클래스의 조건부 확률을 구한다.**


Confidence 계산 방법과 조건부 확률에 대한 수식은 논문을 참조해주세요!


---------

논문에서는 PASCAL VOC detection dataset에 대해 아래와 같이 설정하여 수행했다고 합니다.
  
![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/5f6b4786-ca04-454e-8168-a6d8ddfc9da0)

위의 그림과 같이 CNN의 최종 예측 결과는 7 * 7 * 30 텐서로 나오게 됩니다.

## 2️⃣Network Design
다음으로 CNN 네트워크 디자인 및 특징입니다.
![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/72906095-ce4a-4408-9487-64ebf3473f7e)
(이미지 출처: 논문 원본)


YOLOv1의 네트워크는 24개의 Convolution Layer와 2개의 Fully-Connected Layer로 이루어져 있습니다.


주요 특징은 다음과 같습니다.

**1) Pre-trained Network**
- 이미지의 노란색 부분 즉, 앞부분의 20개의 Convolution Layer는 구글넷을 이용하여 사전에 학습 시킨 모델을 YOLO에 맞게 파인 튜닝한 네트워크입니다.
- 여기에 이미지의 초록색 부분 즉, 4개의 Convolution Layer와 2개의 Fully-Connected Layer를 추가하여 최종 YOLO 네트워크를 구성하였습니다.
  
**2) Reduction Layer**
- 1*1 Reduction Lyaer를 사용하여 연산량을 축소했습니다.
(이 부분에 대해서는 Inception 등의 내용을 추가로 찾아보시기를 추천합니다.)

**3) Activation Function**
- 맨 마지막 Layer에는 Linear를 나머지 Layer에는 Leaky ReLU를 Activation Function으로 사용했습니다.

이러한 CNN 네트워크를 통해 7 * 7 * 30 텐서(PASCAL VOC detection dataset 기준)가 출력되는데요,
이 출력물을 통해 객체의 위치와 클래스를 최종적으로 결정해줘야 합니다.
이때 사용되는 기법이 바로 NMS 입니다.

-------------

**NMS(Non-Maximum Suppression, 비최대 억제)**

: Object Detector가 예측한 Bounding Box 중 정확한 Bounding Box를 선택하도록 하는 기법

NMS의 알고리즘 흐름은 아래와 같습니다.

**1) 모든 Bounding Box에 대하여 threshold 이하의 Confidence score를 가지는 Box는 제거**

**2) 남은 Bounding Box들을 Confidence score 기준으로 내림차순 정렬**

**3) 맨 앞에 있는 Bounding Box 하나를 기준으로, 다른 Bounding Box와 IoU값 계산**
   
   - IoU가 threshold 이상인 Bounding Box는 제거
   - 많이 겹칠수록 같은 부분의 같은 물체를 검출하고 있다고 판단하기 때문!


**4) 위 과정을 순차적, 반복적으로 시행하여 모든 Bounding Box를 비교하고 제거**


이렇게 말로만 적혀있으면 쉽게 이해가 잘 안되실텐데요, 보다 쉽게 설명하고 있는 페이지가 있어 공유합니다.
예제를 통해 이해하고 싶으신 분은 아래 링크를 참조하세요!
- https://wikidocs.net/142645
- https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_749

## 3️⃣Loss Funtion
이번에는 YOLOv1의 Loss Function 입니다.

![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/e7a61d5f-9236-46bb-8fff-30fce935f9d1)

YOLOv1의 Loss Function은 기본적으로 SSE(Sum of Squared Error, 오차제곱합)의 개념을 사용합니다.
이는 예측 결과와 실제 결과의 차이를 제곱하여 총 합을 구한 값을 말합니다.

![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/a2b954b4-5830-4541-a7a4-4d1f813ebdde)

YOLOv1은 SSE의 개념을 사용하되 크게 Localization, Confidence, Class로 나눠서 계산했으며, 객체 포함 여부에 따라 다른 가중치를 주어 계산했습니다. 이때 가중치는 당연히 객체를 인식 부분에 더 큰 값을 주었습니다.

## 4️⃣실험 결과
논문에서는 일반화, 타 Real Time 방법론과의 비교 등 다양한 실험을 수행했습니다.
이 부분은 논문을 참조해주세요!

## 5️⃣결론
YOLOv1은 다음과 같은 **장점**을 가집니다.
- 매우 빠름 (1 stage로 동작)
- 전체 문맥을 보고 판단함 (Region Proposal과 다르게 이미지 전체를 기준으로 판단)
- 일반화 능력이 뛰어남 (사진 데이터 학습 모델을 Artwork에 테스트 했을 때에도 좋은 성능)

반면, 다음과 같은 **한계**를 가집니다.
- 비교적 성능이 떨어짐 (속도는 매우 빠르지만, 성능은 기존 2 stage에 비해 다소 떨어짐)
- 모여있거나 겹쳐있는 물체 검출 성능 떨어짐 (하나의 grid 당 2개의 박스만 검출)

![image](https://github.com/HY-AI2-Projects/YOLOv1_2020071685_Lee-Mijin/assets/146939806/d076e101-5bda-4c53-9813-ec16a7cfce02)
(이미지 출처: https://www.researchgate.net/figure/Timeline-of-You-Only-Look-Once-YOLO-variants_fig1_369379818)

YOLO는 YOLOv1을 시작으로 2023년 현재 YOLOv8 까지 계속해서 발전하고 있으며 아직도 많이 사용되고 있는 모델입니다.
Object Detection을 공부하신다면 YOLO는 필수적으로 학습하시기를 권합니다!

## 💻CODE
YOLOv1 모델 구현과 관련하여 구현이 잘 되어있는 일부 GitHHub 링크를 첨부합니다.

1) https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
2) https://github.com/motokimura/yolo_v1_pytorch
3) https://github.com/JeffersonQin/yolo-v1-pytorch


