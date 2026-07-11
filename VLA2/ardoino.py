import serial
import time

# [설정] 아두이노 포트 및 제어하고 싶은 핀 번호 지정
PORT = 'COM6' 
BAUD_RATE = 9600
SERVO_PIN = 2  # 💡 이제 여기서 2, 7 등 원하는 아두이노 핀 번호로 바꾸면 됩니다!

try:
    # 1. 아두이노와 시리얼 통신 연결
    py_serial = serial.Serial(PORT, BAUD_RATE, timeout=1)
    
    print("아두이노 연결 중입니다. 잠시만 기다려주세요...")
    time.sleep(2) 
    
    current_angle = 0
    
    print("\n==================================================")
    print(f"✅ 아두이노({PORT}) 연결 성공!")
    print(f"🎯 제어 중인 핀: 디지털 {SERVO_PIN}번 핀")
    print(f" 현재 예상 각도: {current_angle}도")
    print("- 0부터 180 사이의 각도를 숫자로 입력하세요.")
    print("- 프로그램을 종료하려면 영어 'q'를 입력하세요.")
    print("==================================================")

    while True:
        user_input = input("\n[입력] 목표 각도: ").strip()

        if user_input.lower() == 'q':
            print("프로그램을 안전하게 종료합니다.")
            break

        if user_input.isdigit():
            target_angle = int(user_input)
            
            if 0 <= target_angle <= 180:
                print(f"🔄 {current_angle}도에서 {target_angle}도까지 5도씩 이동을 시작합니다...")
                
                while current_angle != target_angle:
                    if current_angle < target_angle:
                        current_angle += 5
                        if current_angle > target_angle:
                            current_angle = target_angle
                    else:
                        current_angle -= 5
                        if current_angle < target_angle:
                            current_angle = target_angle
                    
                    # ⭐ [핵심] 아두이노가 읽을 수 있도록 "핀번호,각도\n" 포맷으로 전송합니다.
                    # 예: "7,005\n", "7,010\n"
                    data_to_send = f"{SERVO_PIN},{current_angle}\n"
                    py_serial.write(data_to_send.encode())
                    py_serial.flush()
                    
                    print(f"  -> {SERVO_PIN}번 핀으로 {current_angle:03d}도 명령 전송")
                    time.sleep(0.05) 
                
                print(f"✨ 목표 각도({target_angle}도)에 도달했습니다.")
            else:
                print("❌ [경고] 0에서 180 사이의 각도만 입력할 수 있습니다.")
        else:
            print("❌ [경고] 올바른 숫자를 입력하거나 'q'를 입력해 주세요.")

except serial.SerialException as e:
    print(f"\n❌ [오류] {PORT} 포트를 열 수 없습니다. 시리얼 모니터를 닫아주세요.")
finally:
    if 'py_serial' in locals() and py_serial.is_open:
        py_serial.close()
        print("시리얼 연결이 안전하게 해제되었습니다.")
