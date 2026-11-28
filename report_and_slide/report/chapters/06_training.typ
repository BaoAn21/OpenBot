= Driving Policy Training <ch:training>

== Dataset
All data were recorded on an indoor track marked with blue tape (@fig:track).

recording sessions will include different light condition, such as in morning, in evening 
under room light, afternoon with dim light, etc 


TODO: track dimensions (lane width, total length), recording sessions, forward/backward
split.

#figure(
  grid(
    columns: 4,
    gutter: 6pt,
    image("../figures/track_01.jpg"),
    image("../figures/track_03.jpg"),
    image("../figures/track_04.jpg"),
    image("../figures/track_06.jpg"),
  ),
  caption: [Sections of the indoor track used for data collection and evaluation.],
) <fig:track>

== Loss function
Rare sharp turns would be drowned out by straight driving under a plain MSE, so each
sample is weighted by its steering magnitude (@eq:loss):

$ cal(L) = w^2 (Delta t^2 + 5 Delta s^2), quad w = 2|s| + 0.05 $ <eq:loss>

where $s$ is the target steering, and $Delta s$, $Delta t$ are the steering and
throttle errors. Steering errors cost five times as much as throttle errors.
