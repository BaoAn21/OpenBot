void setup() {

    Serial.begin(115200); // Khởi tạo Serial

    // Cau hinh chan analog
    pinMode(A0, INPUT);
}

void loop() {

    // Doc gia tri cam bien
    int sensorValue = analogRead(A0);

    // In ra gia tri anh sang
    Serial.print("Light: ");
    Serial.println(sensorValue); // In voi 2 chu so thap phan

    delay(1000); // Cho 1 giay
}
