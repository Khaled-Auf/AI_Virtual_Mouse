import cv2 
import mediapipe as mp
import pyautogui
import math



# Hand Tracking 
mp_hands = mp.solutions.hands  
hand = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Get Screen Size
screen_width , screen_height = pyautogui.size()

# Open the Camera
cap = cv2.VideoCapture(0)

# Smoothening Parameters
prev_x , prev_y = 0, 0
smoothening = 5

click_down = False

while True:
    success , img = cap.read()
    if not success:
        break
    
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    
#Get frame size
    frame_height ,frame_width , _ = img.shape
    
    
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hand_lms,mp_hands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand_lms.landmark):
                if id == 8 :
                    
                    # Mapping to frame Coordinates
                    x = int(lm.x * frame_width)
                    y = int(lm.y * frame_height)
                    
                    # Mapping to screen Coordinates + smoothening
                    curr_x = int(lm.x * screen_width)
                    curr_y = int(lm.y * screen_height)
                    screen_x = prev_x + (curr_x - prev_x) // smoothening
                    screen_y = prev_y + (curr_y - prev_y) // smoothening
                    
                    # Move Mouse to Mapped Position after Smoothening
                    pyautogui.moveTo(screen_x,screen_y)
                    prev_x , prev_y = screen_x , screen_y
                    cv2.circle(img,(x,y),10,(255,0,255),cv2.FILLED)
                    
                    thumb =  hand_lms.landmark[4]
                    thumb_x = int(thumb.x * frame_width)
                    thumb_y = int(thumb.y * frame_height)
                    
                    distance = math.hypot(x-thumb_x , y-thumb_y)
                    
                    if distance < 30:
                        if not click_down:
                            pyautogui.click()
                            click_down = True
                            cv2.circle(img,(x,y),15,(0,255,0),cv2.FILLED)
                    else:
                        click_down = False
                    
            
    cv2.imshow("Virtual Mouse",img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


