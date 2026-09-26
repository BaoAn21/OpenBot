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
int indicator_reverse = 0;

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
  tmp = strtok(NULL, ",:");  // optional third field: reverse light
  indicator_reverse = tmp == NULL ? 0 : atoi(tmp);
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
// The screen faces backwards on the rear of the car and doubles as the turn signals:
// sequential amber chevrons on each side, steering/throttle gauges in the middle.

#define SCREEN_W 320
#define SCREEN_H 170

#define COLOR_BG RGB565_BLACK
#define COLOR_PANEL RGB565(24, 24, 32)
#define COLOR_FRAME RGB565(70, 70, 90)
#define COLOR_LABEL RGB565(150, 150, 170)
#define COLOR_AMBER RGB565(255, 150, 0)
#define COLOR_AMBER_DIM RGB565(45, 26, 0)
#define COLOR_STEER RGB565(0, 200, 255)
#define COLOR_FWD RGB565(0, 230, 110)
#define COLOR_REV RGB565(255, 90, 40)
#define COLOR_REVERSE_DIM RGB565(60, 60, 70)

// Indicator zones on the left and right edges; the gauges live in between.
#define IND_W 78
#define CENTER_X IND_W
#define CENTER_W (SCREEN_W - 2 * IND_W)

// Three chevrons per side, lit one after another like a sequential turn signal.
#define CHEVRONS 3
#define CHEV_DX 24       // horizontal reach of each arm
#define CHEV_DY 52       // vertical reach of each arm
#define CHEV_T 10        // stroke thickness
#define CHEV_SPACING 17  // distance between chevrons
#define BLINK_PERIOD_MS 800
#define BLINK_STEP_MS 140  // delay before the next chevron lights up
#define BLINK_ON_MS 500    // all chevrons go dark after this, until the period ends

#define GAUGE_X (CENTER_X + 8)
#define GAUGE_W (CENTER_W - 16)
#define BAR_H 16

unsigned long blink_start = 0;

// Chevron pointing left (dir = -1) or right (dir = +1) with its tip at (tip_x, cy).
void draw_chevron(int tip_x, int cy, int dir, uint16_t color) {
  int arm_x = tip_x - dir * CHEV_DX;
  int in_tip = tip_x - dir * CHEV_T;
  int in_arm = arm_x - dir * CHEV_T;
  gfx->fillTriangle(tip_x, cy, arm_x, cy - CHEV_DY, in_arm, cy - CHEV_DY, color);
  gfx->fillTriangle(tip_x, cy, in_arm, cy - CHEV_DY, in_tip, cy, color);
  gfx->fillTriangle(tip_x, cy, arm_x, cy + CHEV_DY, in_arm, cy + CHEV_DY, color);
  gfx->fillTriangle(tip_x, cy, in_arm, cy + CHEV_DY, in_tip, cy, color);
}

// Number of chevrons lit at this point of the blink cycle (outermost lights last).
int lit_chevrons(unsigned long now) {
  unsigned long t = (now - blink_start) % BLINK_PERIOD_MS;
  if (t >= BLINK_ON_MS) return 0;
  return min(CHEVRONS, (int)(t / BLINK_STEP_MS) + 1);
}

// Chevron 0 sits next to the gauges and lights first; the rest step out towards the edge.
void draw_indicator(int dir, int lit) {
  int cy = SCREEN_H / 2;
  int inner_tip = dir < 0 ? IND_W - 6 - CHEV_DX - CHEV_T : SCREEN_W - IND_W + 6 + CHEV_DX + CHEV_T;
  for (int i = 0; i < CHEVRONS; i++) {
    int tip_x = inner_tip + dir * i * CHEV_SPACING;
    draw_chevron(tip_x, cy, dir, i < lit ? COLOR_AMBER : COLOR_AMBER_DIM);
  }
}

// Horizontal bar centred at zero: fills right for positive values, left for negative.
void draw_bar(int y, int value, uint16_t color) {
  int mid = GAUGE_X + GAUGE_W / 2;
  int len = value * (GAUGE_W / 2 - 2) / CTRL_MAX;
  gfx->fillRoundRect(GAUGE_X, y, GAUGE_W, BAR_H, 4, COLOR_BG);
  gfx->drawRoundRect(GAUGE_X, y, GAUGE_W, BAR_H, 4, COLOR_FRAME);
  if (len > 0) gfx->fillRect(mid, y + 3, len, BAR_H - 6, color);
  if (len < 0) gfx->fillRect(mid + len, y + 3, -len, BAR_H - 6, color);
  gfx->drawFastVLine(mid, y + 1, BAR_H - 2, RGB565_WHITE);
}

void draw_gauge(int y, const char *label, int value, uint16_t color) {
  char buf[8];
  snprintf(buf, sizeof(buf), "%4d", value);
  gfx->setTextSize(1);
  gfx->setTextColor(COLOR_LABEL, COLOR_PANEL);
  gfx->setCursor(GAUGE_X, y + 6);
  gfx->print(label);
  gfx->setTextSize(2);
  gfx->setTextColor(RGB565_WHITE, COLOR_PANEL);
  gfx->setCursor(GAUGE_X + GAUGE_W - 48, y);
  gfx->print(buf);
  draw_bar(y + 20, value, color);
}

// White badge at the top of the panel, lit while reversing (like a reversing light).
void draw_reverse(bool on) {
  int x = GAUGE_X, y = 8, w = GAUGE_W, h = 24;
  gfx->fillRoundRect(x, y, w, h, 6, on ? RGB565_WHITE : COLOR_PANEL);
  gfx->drawRoundRect(x, y, w, h, 6, on ? RGB565_WHITE : COLOR_REVERSE_DIM);
  gfx->setTextSize(2);
  gfx->setTextColor(on ? RGB565_BLACK : COLOR_REVERSE_DIM);
  gfx->setCursor(x + (w - 7 * 12) / 2, y + 5);
  gfx->print("REVERSE");
}

void draw_link(bool ok) {
  int y = SCREEN_H - 22;
  gfx->fillRect(CENTER_X + 4, y - 2, CENTER_W - 8, 18, COLOR_PANEL);
  gfx->fillCircle(GAUGE_X + 6, y + 7, 5, ok ? COLOR_FWD : RGB565_RED);
  gfx->setTextSize(2);
  gfx->setTextColor(ok ? COLOR_FWD : RGB565_RED, COLOR_PANEL);
  gfx->setCursor(GAUGE_X + 18, y);
  gfx->print(ok ? "LINK" : "NO LINK");
}

void draw_static() {
  gfx->fillScreen(COLOR_BG);
  gfx->fillRoundRect(CENTER_X, 2, CENTER_W, SCREEN_H - 4, 10, COLOR_PANEL);
  gfx->drawRoundRect(CENTER_X, 2, CENTER_W, SCREEN_H - 4, 10, COLOR_FRAME);
  draw_indicator(-1, 0);
  draw_indicator(1, 0);
}

void update_display() {
  static int last_steering = INT_MIN, last_throttle = INT_MIN;
  static int last_lit_left = -1, last_lit_right = -1;
  static int last_link = -1;
  static int last_reverse = -1;
  static bool was_blinking = false;

  unsigned long now = millis();
  bool blinking = indicator_left || indicator_right;
  if (blinking && !was_blinking) blink_start = now;  // start each blink with the first chevron lit
  was_blinking = blinking;

  int lit = blinking ? lit_chevrons(now) : 0;
  int lit_left = indicator_left ? lit : 0;
  int lit_right = indicator_right ? lit : 0;
  if (lit_left != last_lit_left) {
    draw_indicator(-1, lit_left);
    last_lit_left = lit_left;
  }
  if (lit_right != last_lit_right) {
    draw_indicator(1, lit_right);
    last_lit_right = lit_right;
  }

  if (ctrl_steering != last_steering) {
    draw_gauge(40, "STEER", ctrl_steering, COLOR_STEER);
    last_steering = ctrl_steering;
  }
  if (ctrl_throttle != last_throttle) {
    draw_gauge(90, "THROTTLE", ctrl_throttle, ctrl_throttle >= 0 ? COLOR_FWD : COLOR_REV);
    last_throttle = ctrl_throttle;
  }
  if (indicator_reverse != last_reverse) {
    draw_reverse(indicator_reverse);
    last_reverse = indicator_reverse;
  }
  if (link_ok != last_link) {
    draw_link(link_ok);
    last_link = link_ok;
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
    indicator_left = 0;  // don't keep signalling a turn the phone no longer controls
    indicator_right = 0;
    indicator_reverse = 0;
  }

  write_steering(ctrl_steering);
  write_throttle(ctrl_throttle);

  static unsigned long display_time = 0;
  if (millis() - display_time >= 20) {  // fast enough for smooth chevron steps
    update_display();
    display_time = millis();
  }
}
