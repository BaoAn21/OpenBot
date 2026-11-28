#import "template.typ": thesis

#show: thesis.with(
  title: "Deep Learning Application on Smartphones for Autonomous Vehicles",
  thesis-type: "GRADUATION THESIS",
  kind: "THESIS",
  major: "Computer Science",
  supervisors: ("Dr. Le Trong Nhan",),
  students: (("Tran Cao Bao An", "2370297"),),
  place-date: "Ho Chi Minh City, December 2027",
  commitment: [
    I pledge that this thesis is based on my own work under the guidance of my
    supervisor. All studies, data, references, numbers, and statistics presented here
    are reliable and honestly cited. The work meets the requirements set by the Faculty
    of Computer Science and Engineering.
  ],
  acknowledgement: [
    First and foremost, I would like to express my deepest gratitude to my thesis
    supervisor, *Dr. Le Trong Nhan*, for the invaluable guidance, encouragement, and support
    throughout the course of this work.

    I sincerely thank the lecturers of the *Faculty of Computer Science and
    Engineering* at *Ho Chi Minh City University of Technology* for the knowledge and
    experience they have shared with me during my study.
  ],
  abstract: [
    _Motivation._ TODO: why a learned autopilot on a low-cost car, and what was
    missing from the Part 2 OpenCV approach.

    _Proposed approach._ TODO: OpenBot on an Ackermann RC car (phone + ESP32-S3), with a
    memory-based driving model integrated into its training pipeline and app.

    _Methodology._ TODO: data collection, models compared, training choices, evaluation
    on the track.

    _Results._ TODO: main measured result.
  ],
  keywords: ("smartphone", "RC car", "imitation learning", "OpenBot", "autopilot", "TFLite", "ESP32"),
  abbreviations: (
    ("AI", "Artificial Intelligence"),
    ("BC", "Behavioural Cloning"),
    ("BN", "Batch Normalisation"),
    ("CIL", "Conditional Imitation Learning"),
    ("CNN", "Convolutional Neural Network"),
    ("ESC", "Electronic Speed Controller"),
    ("FPS", "Frames Per Second"),
    ("GPU", "Graphics Processing Unit"),
    ("MAE", "Mean Absolute Error"),
    ("MCU", "Microcontroller Unit"),
    ("ML", "Machine Learning"),
    ("MSE", "Mean Squared Error"),
    ("PWM", "Pulse-Width Modulation"),
    ("RC", "Radio-Controlled"),
    ("TFLite", "TensorFlow Lite"),
    ("UART", "Universal Asynchronous Receiver-Transmitter"),
    ("USB", "Universal Serial Bus"),
  ),
)

#include "chapters/01_introduction.typ"
#include "chapters/02_background.typ"
#include "chapters/03_hardware.typ"
#include "chapters/04_firmware.typ"
#include "chapters/05_android_app.typ"
#include "chapters/06_dataset.typ"
#include "chapters/07_models.typ"
#include "chapters/08_results.typ"
#include "chapters/09_conclusion.typ"

#bibliography("references.bib", title: "References", style: "ieee")
