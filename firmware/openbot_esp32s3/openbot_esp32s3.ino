// OpenBot firmware for the LilyGO T-Display-S3 (ESP32-S3).
//
// Ackermann-style car: a steering servo plus one drive motor.
// The phone sends "c<steering>,<throttle>\n", both in [-255, 255]
// (positive steering = right, positive throttle = forward).
//
// Board settings (arduino-cli):
//   esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,FlashSize=16M,PSRAM=opi,PartitionScheme=app3M_fat9M_16MB
// Library: "GFX Library for Arduino"

#include <Arduino_GFX_Library.h>

//------------------------------------------------------//
// CONFIGURATION
//------------------------------------------------------//

// Motor driver type. Only HBRIDGE is implemented for now; ESC comes next.
#define HBRIDGE 0
#define ESC 1
#define MOTOR_DRIVER HBRIDGE

// Reported to the app as the robot type (shown in Robot Info).
const String robot_type = "TDISPLAY_S3";

// Free GPIOs on the T-Display-S3 headers: 1, 2, 3, 10, 11, 12, 13, 16, 17, 18, 21, 43, 44.
// Everything else is used by the display, buttons or battery sense.
#define PIN_STEERING_SERVO 10

// H-bridge wired as two PWM inputs (IN1/IN2). Works with L298N (ENA jumpered),
// DRV8833, MX1508, TB6612 (PWMA tied high), BTS7960 (RPWM/LPWM, R_EN/L_EN tied high).
#define PIN_MOTOR_IN1 1
#define PIN_MOTOR_IN2 2

// Servo pulse widths in microseconds. Tune these so the wheels don't hit the end stops.
#define SERVO_MIN_US 1000     // full left
#define SERVO_CENTER_US 1500  // straight
#define SERVO_MAX_US 2000     // full right
#define STEERING_INVERT false // set true if the wheels turn the wrong way

#define THROTTLE_INVERT false // set true if the car drives backwards on forward
#define MOTOR_MIN_PWM 0       // smallest PWM that actually moves the motor (overcomes stiction)
#define MOTOR_BRAKE false     // at zero throttle: true = short brake, false = coast

// Stop the car if no heartbeat arrives within the interval the phone asked for.
#define DEFAULT_HEARTBEAT_MS 1000

//------------------------------------------------------//
// HARDWARE CONSTANTS
//------------------------------------------------------//

#define CTRL_MAX 255

#define SERVO_FREQ_HZ 50
#define SERVO_RES_BITS 14
#define SERVO_PERIOD_US (1000000 / SERVO_FREQ_HZ)

#define MOTOR_FREQ_HZ 20000  // above hearing range, no motor whine
#define MOTOR_RES_BITS 8     // duty 0..255 matches CTRL_MAX

#define PIN_LCD_POWER 15
#define PIN_LCD_BL 38

Arduino_DataBus *bus = new Arduino_ESP32LCD8(
    7 /* DC */, 6 /* CS */, 8 /* WR */, 9 /* RD */,
    39, 40, 41, 42, 45, 46, 47, 48 /* D0..D7 */);
Arduino_GFX *gfx = new Arduino_ST7789(
    bus, 5 /* RST */, 1 /* rotation: landscape */, true /* IPS */, 170, 320,
    35, 0, 35, 0 /* col/row offsets */);

//------------------------------------------------------//
// STATE
//------------------------------------------------------//

int ctrl_steering = 0;
int ctrl_throttle = 0;
int indicator_left = 0;
int indicator_right = 0;

unsigned long heartbeat_interval = DEFAULT_HEARTBEAT_MS;
unsigned long heartbeat_time = 0;
bool link_ok = false;

char msg_buf[64];
int msg_idx = 0;
char header = 0;

//------------------------------------------------------//
// ACTUATORS
//------------------------------------------------------//

void write_steering(int steering) {
  if (STEERING_INVERT) steering = -steering;
  int us = steering >= 0
               ? map(steering, 0, CTRL_MAX, SERVO_CENTER_US, SERVO_MAX_US)
               : map(steering, -CTRL_MAX, 0, SERVO_MIN_US, SERVO_CENTER_US);
  uint32_t duty = (uint32_t)us * ((1 << SERVO_RES_BITS) - 1) / SERVO_PERIOD_US;
  ledcWrite(PIN_STEERING_SERVO, duty);
}

#if (MOTOR_DRIVER == HBRIDGE)
void write_throttle(int throttle) {
  if (THROTTLE_INVERT) throttle = -throttle;
  int pwm = abs(throttle);
  if (pwm == 0) {
    int idle = MOTOR_BRAKE ? CTRL_MAX : 0;
    ledcWrite(PIN_MOTOR_IN1, idle);
    ledcWrite(PIN_MOTOR_IN2, idle);
    return;
  }
  pwm = map(pwm, 1, CTRL_MAX, MOTOR_MIN_PWM, CTRL_MAX);
  if (throttle > 0) {
    ledcWrite(PIN_MOTOR_IN1, pwm);
    ledcWrite(PIN_MOTOR_IN2, 0);
  } else {
    ledcWrite(PIN_MOTOR_IN1, 0);
    ledcWrite(PIN_MOTOR_IN2, pwm);
  }
}
#endif

void setup_actuators() {
  ledcAttach(PIN_STEERING_SERVO, SERVO_FREQ_HZ, SERVO_RES_BITS);
#if (MOTOR_DRIVER == HBRIDGE)
  ledcAttach(PIN_MOTOR_IN1, MOTOR_FREQ_HZ, MOTOR_RES_BITS);
  ledcAttach(PIN_MOTOR_IN2, MOTOR_FREQ_HZ, MOTOR_RES_BITS);
#endif
  write_steering(0);
  write_throttle(0);
}

//------------------------------------------------------//
// SERIAL PROTOCOL
//------------------------------------------------------//

void process_ctrl_msg() {
  char *tmp = strtok(msg_buf, ",:");
  if (tmp == NULL) return;
  int steering = atoi(tmp);
  tmp = strtok(NULL, ",:");
  if (tmp == NULL) return;
  int throttle = atoi(tmp);
  ctrl_steering = constrain(steering, -CTRL_MAX, CTRL_MAX);
  ctrl_throttle = constrain(throttle, -CTRL_MAX, CTRL_MAX);
}

void process_heartbeat_msg() {
  heartbeat_interval = atol(msg_buf);
  heartbeat_time = millis();
}

void process_indicator_msg() {
  char *tmp = strtok(msg_buf, ",:");
  if (tmp == NULL) return;
  indicator_left = atoi(tmp);
  tmp = strtok(NULL, ",:");
  if (tmp == NULL) return;
  indicator_right = atoi(tmp);
}

void process_feature_msg() {
  // No optional sensors yet; features get appended here as "v:", "i:", "wf:", ...
  Serial.println("f" + robot_type + ":");
}

void parse_msg() {
  switch (header) {
    case 'c':
      process_ctrl_msg();
      break;
    case 'h':
      process_heartbeat_msg();
      break;
    case 'i':
      process_indicator_msg();
      break;
    case 'f':
      process_feature_msg();
      break;
  }
  msg_idx = 0;
  header = 0;
}

void on_serial_rx() {
  char c = Serial.read();
  if (c == '\r') return;
  if (c == '\n') {
    msg_buf[msg_idx] = '\0';
    parse_msg();
  } else if (header == 0) {
    header = c;
  } else if (msg_idx < (int)sizeof(msg_buf) - 1) {
    msg_buf[msg_idx++] = c;
  }
}

//------------------------------------------------------//
// DISPLAY
//------------------------------------------------------//

#define BAR_X 110
#define BAR_W 200
#define BAR_H 22

// Horizontal bar centred at zero: fills right for positive values, left for negative.
void draw_bar(int y, int value, uint16_t color) {
  int mid = BAR_X + BAR_W / 2;
  int len = value * (BAR_W / 2) / CTRL_MAX;
  gfx->fillRect(BAR_X, y, BAR_W, BAR_H, RGB565_BLACK);
  gfx->drawRect(BAR_X, y, BAR_W, BAR_H, RGB565_DARKGREY);
  if (len > 0) gfx->fillRect(mid, y + 2, len, BAR_H - 4, color);
  if (len < 0) gfx->fillRect(mid + len, y + 2, -len, BAR_H - 4, color);
  gfx->drawFastVLine(mid, y, BAR_H, RGB565_WHITE);
}

void draw_value(int y, const char *label, int value) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%-5s%4d", label, value);
  gfx->setTextSize(2);
  gfx->setTextColor(RGB565_WHITE, RGB565_BLACK);
  gfx->setCursor(4, y + 4);
  gfx->print(buf);
}

void draw_static() {
  gfx->fillScreen(RGB565_BLACK);
  gfx->setTextSize(3);
  gfx->setTextColor(RGB565_GREEN);
  gfx->setCursor(4, 4);
  gfx->print("OpenBot S3");
}

void update_display() {
  static int last_steering = INT_MIN, last_throttle = INT_MIN;
  static int last_ind_left = -1, last_ind_right = -1;
  static int last_link = -1;

  if (link_ok != last_link) {
    gfx->setTextSize(2);
    gfx->setCursor(210, 8);
    gfx->setTextColor(link_ok ? RGB565_GREEN : RGB565_RED, RGB565_BLACK);
    gfx->print(link_ok ? "  LINK" : "NO HB ");
    last_link = link_ok;
  }
  if (ctrl_steering != last_steering) {
    draw_value(50, "STR", ctrl_steering);
    draw_bar(50, ctrl_steering, RGB565_CYAN);
    last_steering = ctrl_steering;
  }
  if (ctrl_throttle != last_throttle) {
    draw_value(90, "THR", ctrl_throttle);
    draw_bar(90, ctrl_throttle, ctrl_throttle >= 0 ? RGB565_GREEN : RGB565_ORANGE);
    last_throttle = ctrl_throttle;
  }
  if (indicator_left != last_ind_left || indicator_right != last_ind_right) {
    gfx->setTextSize(3);
    gfx->setCursor(4, 135);
    gfx->setTextColor(indicator_left ? RGB565_YELLOW : RGB565_DARKGREY, RGB565_BLACK);
    gfx->print("<");
    gfx->setCursor(296, 135);
    gfx->setTextColor(indicator_right ? RGB565_YELLOW : RGB565_DARKGREY, RGB565_BLACK);
    gfx->print(">");
    last_ind_left = indicator_left;
    last_ind_right = indicator_right;
  }
}

void setup_display() {
  pinMode(PIN_LCD_POWER, OUTPUT);
  digitalWrite(PIN_LCD_POWER, HIGH);  // panel is unpowered until this is HIGH
  gfx->begin();
  draw_static();
  pinMode(PIN_LCD_BL, OUTPUT);
  digitalWrite(PIN_LCD_BL, HIGH);
}

//------------------------------------------------------//
// MAIN
//------------------------------------------------------//

void setup() {
  setup_actuators();  // first, so the car is stopped/centred as early as possible
  Serial.begin(115200);
  setup_display();
}

void loop() {
  while (Serial.available()) on_serial_rx();

  link_ok = heartbeat_time != 0 && millis() - heartbeat_time <= heartbeat_interval;
  if (!link_ok) {
    ctrl_steering = 0;
    ctrl_throttle = 0;
  }

  write_steering(ctrl_steering);
  write_throttle(ctrl_throttle);

  static unsigned long display_time = 0;
  if (millis() - display_time >= 50) {  // 20 fps is plenty
    update_display();
    display_time = millis();
  }
}
