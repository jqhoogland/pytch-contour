# pytch-contour
Estimate pitch contours from spoken clips with python. For practice with tonal languages like Mandarin.

This is based off of a [paper by Paul Boersma (1993)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.4956&rep=rep1&type=pdf). The methodology is the same as is in his software, [Praat](https://github.com/praat/praat), but implemented on python. Rather than focus on linguistic value, I'd like this to be pedagogical: for practicing intonation.

### Usage

To record and analyze a sample of your own voice:

```pipenv run python process.py```

##### Me saying "Hello world" 
![hello-world.png](/docs/hello-world.png)

To analyze prerecorded sample into a contour:

```pipenv run python process.py -p audio/mandarin/zhong4yao4.wav```

##### "Zhòng yào" (I don't speak any Mandarin and have no idea what this means)
![zhong4yao4.png](/docs/zhong4yao4.png)


To also play the corresponding audio, use the `-a, --audio` flag.

To play a prerecorded sample, then record your imitation, then compare the two:

##### "Zhòng yào" (Guess who's who)
```pipenv run python process.py -p audio/mandarin/zhong4yao4.wav -r```

![zhong4yao4-comparison.png](/docs/zhong4yao4-comparison.png)


### Options
By default, the options are the same as in Boersma's original paper. 

```
  -P PITCH_RANGE PITCH_RANGE, --pitch-range PITCH_RANGE PITCH_RANGE
                        Default: [75, 500]
  -O OCTAVE_COST, --octave-cost OCTAVE_COST
                        Default: 0.01
  -V VOICING_THRESHOLD, --voicing-threshold VOICING_THRESHOLD
                        Default: 0.45
  -S SILENCE_THRESHOLD, --silence-threshold SILENCE_THRESHOLD
                        Default: 0.03
  -J OCTAVE_JUMP_COST, --octave-jump-cost OCTAVE_JUMP_COST
                        Default: 0.35
  -T VOICING_TRANSITION_COST, --voicing-transition-cost VOICING_TRANSITION_COST
                        Default: 0.14
```

You might have to play around with these settings to get something that works for you. I like `silence_threshold=0.15` (I don't record my audio in a studio) and `octave_jump_cost=0.4`

 
