#include <Servo.h>

Servo myServo;
int currentPin = -1; // 현재 서보모터가 연결된 핀 번호를 저장할 변수

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

    // [핀 변경 처리] 이전에 연결된 핀과 다르다면 새로 attach 해줍니다.
    if (pinNumber != currentPin && pinNumber > 0) {
      myServo.detach();           // 기존 핀 연결 해제
      myServo.attach(pinNumber);   // 새로운 핀 연결
      currentPin = pinNumber;     // 현재 핀 번호 갱신
    }

    // [모터 구동] 각도가 올바른 범위라면 모터를 움직입니다.
    if (angle >= 0 && angle <= 180 && currentPin != -1) {
      myServo.write(angle);
    }
  }
}
