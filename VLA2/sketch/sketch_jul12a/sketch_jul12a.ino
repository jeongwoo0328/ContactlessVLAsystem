#include <Servo.h>

// 동적으로 연결할 4개의 서보모터 객체 선언
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;

// 각각의 서보모터가 현재 연결된 핀 번호를 저장할 변수 (-1은 연결 안 됨을 뜻함)
int pin1 = -1;
int pin2 = -1;
int pin3 = -1;
int pin4 = -1;

// 4개의 슬롯이 모두 꽉 찼을 때, 새로 들어온 핀을 교체할 순서를 기억하는 변수 (0, 1, 2, 3 순환)
int replaceIndex = 0;

void setup() {
  Serial.begin(9600); // 파이썬과 통신 속도 맞춤
}

void loop() {
  // 시리얼 포트에 데이터가 들어왔는지 확인
  if (Serial.available() > 0) {
    
    // 1. 첫 번째 숫자를 읽음 (핀 번호)
    int pinNumber = Serial.parseInt(); 
    
    // 2. 그 다음 숫자를 읽음 (목표 각도)
    int angle = Serial.parseInt(); 
    
    // 3. 데이터의 끝(\n)까지 시리얼 버퍼를 비워줍니다.
    while (Serial.available() > 0 && Serial.peek() == '\n') {
      Serial.read();
    }

    // [각도 유효성 검사] 각도가 올바른 범위(0~180)이고 핀 번호가 올바를 때만 작동합니다.
    if (angle >= 0 && angle <= 180 && pinNumber > 0) {
      
      // [상태 1] 이미 해당 핀이 연결되어 있는 경우 -> 기존 서보모터의 각도만 조절합니다.
      if (pinNumber == pin1) {
        servo1.write(angle);
      } 
      else if (pinNumber == pin2) {
        servo2.write(angle);
      } 
      else if (pinNumber == pin3) {
        servo3.write(angle);
      } 
      else if (pinNumber == pin4) {
        servo4.write(angle);
      } 
      // [상태 2] 완전히 새로운 핀이 들어온 경우
      else {
        // 아직 연결되지 않은 빈 슬롯이 있다면 거기에 우선 할당합니다.
        if (pin1 == -1) {
          servo1.attach(pinNumber);
          pin1 = pinNumber;
          servo1.write(angle);
        } 
        else if (pin2 == -1) {
          servo2.attach(pinNumber);
          pin2 = pinNumber;
          servo2.write(angle);
        } 
        else if (pin3 == -1) {
          servo3.attach(pinNumber);
          pin3 = pinNumber;
          servo3.write(angle);
        } 
        else if (pin4 == -1) {
          servo4.attach(pinNumber);
          pin4 = pinNumber;
          servo4.write(angle);
        } 
        // [상태 3] 4개 채널이 모두 가동 중일 때 새로운 핀이 들어온 경우 -> 차례대로 하나를 끊고 새로 연결합니다.
        else {
          if (replaceIndex == 0) {
            servo1.detach();           // 기존 핀 끊기
            servo1.attach(pinNumber);   // 새 핀 연결하기
            pin1 = pinNumber;           // 정보 갱신
            servo1.write(angle);        // 동작
            replaceIndex = 1;           // 다음 교체 대상 지정
          } 
          else if (replaceIndex == 1) {
            servo2.detach();
            servo2.attach(pinNumber);
            pin2 = pinNumber;
            servo2.write(angle);
            replaceIndex = 2;
          } 
          else if (replaceIndex == 2) {
            servo3.detach();
            servo3.attach(pinNumber);
            pin3 = pinNumber;
            servo3.write(angle);
            replaceIndex = 3;
          }
          else if (replaceIndex == 3) {
            servo4.detach();
            servo4.attach(pinNumber);
            pin4 = pinNumber;
            servo4.write(angle);
            replaceIndex = 0;           // 다시 첫 번째 슬롯으로 순환
          }
        }
      }
    }
  }
}