= Driving Policy Training <ch:training>

== Dataset
TODO: recording sessions, forward/backward split.

== Loss function
Rare sharp turns would be drowned out by straight driving under a plain MSE, so each
sample is weighted by its steering magnitude (@eq:loss):

$ cal(L) = w^2 (Delta t^2 + 5 Delta s^2), quad w = 2|s| + 0.05 $ <eq:loss>

where $s$ is the target steering, and $Delta s$, $Delta t$ are the steering and
throttle errors. Steering errors cost five times as much as throttle errors.
