int ledCat = 12; // LED للقط
int ledDog = 13; // LED للكلب

void setup() {
  pinMode(ledCat, OUTPUT);
  pinMode(ledDog, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();

    if (data == 'C') {
      digitalWrite(ledCat, HIGH);
      digitalWrite(ledDog, LOW);
    }
    else if (data == 'D') {
      digitalWrite(ledCat, LOW);
      digitalWrite(ledDog, HIGH);
    }
    else if (data == '0') {
      digitalWrite(ledCat, LOW);
      digitalWrite(ledDog, LOW);
    }
  }
}
