= Experiments and Results <ch:results>

This chapter evaluates the two driving policies of @ch:training, CIL-Mobile-Fast and the
recurrent model, first offline on the recorded test data and then closed loop on the
track. The closed-loop evaluation follows the protocol of OpenBot @muller2021openbot,
adapted to the track and manoeuvres of this work.

== Offline evaluation <sec:offline-eval>
Each model is first evaluated on the test split of the dataset (@ch:dataset), which
contains only recordings that were not used for training. In addition to the mean absolute
error of the steering and throttle, two metrics proposed by OpenBot @muller2021openbot are
reported. The first is the fraction of frames whose predicted steering is within 0.1 of
the target steering. The second, less strict, is the fraction of frames whose predicted
steering has the same direction as the target. OpenBot reports that these metrics are more
reliable than the validation loss, but that their correlation with the final driving
performance is still weak @muller2021openbot. Offline results are therefore reported for
completeness, and the comparison of the models rests on the closed-loop evaluation of
@sec:closed-loop.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, center, center),
    table.header(
      [*Model*], [*Driving*\ *direction*], [*Steering*\ *within 0.1*], [*Steering*\ *direction*],
      [*MAE*],
    ),
    [CIL-Mobile-Fast], [Forward], [—], [—], [—],
    [CIL-Mobile-Fast], [Backward], [—], [—], [—],
    [Recurrent model], [Forward], [—], [—], [—],
    [Recurrent model], [Backward], [—], [—], [—],
  ),
  caption: [Offline evaluation on the test split. MAE is the mean absolute error of the
    normalised steering and throttle.],
) <tab:offline>

TODO: the sessions each model was trained on and the size of the test split.

== Closed-loop evaluation <sec:closed-loop>

=== Protocol
OpenBot evaluates its driving policies at a T-junction: one trial consists of six segments,
two straights, two left turns and two right turns, and each policy is evaluated over three
trials @muller2021openbot. The same structure is used here, with segments taken from the
manoeuvres of this work (@fig:track-paths). One trial consists of six segments:

- *Forward straight* (2 segments): driving forward along the corridor, once in each
  direction, with command 0.
- *Forward right turn* (2 segments): driving forward along the corridor and turning right
  into a side branch, once into each branch, with command 1.
- *Backward* (2 segments): reversing out of a side branch into the corridor, once from each
  branch, with command 2.

Each segment starts with the car at a standstill at a marked start position and ends at a
marked end position. The command for the segment is set before the start. Each policy is
evaluated as a pair of a forward and a backward model of the same architecture, on the
Autopilot Session screen (@sec:app-session), over three trials.

All trials are run under the same conditions: the same phone and car, a fully charged
battery at the start of each trial, the same time of day and lighting, and the _fixed_
throttle mode, so that both policies drive at the same speed and differ only in their
steering. Every run is recorded with the screen's logging enabled.

TODO: confirm the fixed throttle value (device units), the time of day and lighting
condition, and the start and end positions of each segment.

=== Metrics
The metrics follow OpenBot @muller2021openbot, with one adaptation for the track:

- *Success.* A binary value per segment, indicating whether the segment was completed. A
  segment fails when the operator has to take over: when the car leaves the track, when it
  does not follow the command (for example, when it drives straight past the branch it was
  told to turn into), or when it stops before the end of the segment.
- *Distance.* The distance driven before the segment failed, as a percentage of the segment
  length; a completed segment counts as 100%. As in OpenBot, a segment whose turn is missed
  is credited only up to the start of the junction.
- *Boundary crossings.* The number of times a wheel crosses the boundary tape within a
  segment without the segment failing. This replaces OpenBot's count of collisions, as the
  track is bounded by tape on the floor rather than by walls.
- *Inference speed.* The average number of frames processed by the model per second,
  computed from the timestamps of the processed frames in the run's log.

As in OpenBot, the metrics are averaged over the six segments of a trial, and the mean and
standard deviation over the three trials are reported.

=== Results
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header(
      [*Policy*], [*Distance* ↑], [*Success* ↑], [*Boundary crossings* ↓], [*FPS* ↑],
    ),
    [CIL-Mobile-Fast], [—], [—], [—], [—],
    [Recurrent model], [—], [—], [—], [—],
  ),
  caption: [Closed-loop evaluation, mean ± standard deviation over three trials of six
    segments each.],
) <tab:closed-loop>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header(
      [*Policy*], [*Forward straight*], [*Forward right turn*], [*Backward*],
    ),
    [CIL-Mobile-Fast], [— / 6], [— / 6], [— / 6],
    [Recurrent model], [— / 6], [— / 6], [— / 6],
  ),
  caption: [Successful segments per segment type, summed over the three trials.],
) <tab:segment-success>

TODO: results.

== Discussion
TODO: compare the two policies; relate the offline metrics to the closed-loop results;
typical failure cases.
