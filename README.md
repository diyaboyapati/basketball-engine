# basketball-engine

A real-time basketball play-by-play engine. It reads game events as they arrive,
keeps the score and box score up to date, and can quickly answer questions about
any stretch of the game — like "who scored more between minutes 18 and 24?" or
"what was the biggest run, and when did it happen?"

No outside libraries. Python 3.10+.

```bash
git clone https://github.com/YOUR_USERNAME/basketball-engine.git
cd basketball-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest
bball replay --synthetic
bball query --synthetic --start-minute 18 --end-minute 24
```

## The problem in plain words

Imagine the system behind a live scoreboard. A feed sends you a stream of events. A made three, a rebound, a foul, a turnover. You have to keep everything current as they come in.

Two things make this harder than it sounds.

**Events do not arrive in order.** Network delays mean an event from ten seconds ago can show up after one from five seconds ago. You cannot just sort them because the stream has not finished. More events are still coming.

**Some questions are slow to answer.** Points in the third quarter sounds easy, but if you answer it by looping through every event in the game then it gets slower as the game goes on. During a live broadcast where you might be asked constantly, that is a problem.

This project solves both. The ordering problem gets a buffer that holds events briefly and releases them in the right sequence. The query problem gets two data structures that answer questions about any time window almost instantly, no matter how long the game has run.

## What I set out to learn

I wanted to build a **Fenwick tree** and a **segment tree** from scratch. No libraries and no copying. I wanted to understand not just how they work but when you need which one.

Most explanations present them as two ways to do the same job. Building both against a real problem showed me they are not interchangeable. The reason turned out to be the most interesting thing I learned here.

The rest of the project exists to give those structures something real to work on.

## How fast is it

The game is about 2,880 seconds long. Say you want points scored in a six minute window.

**The obvious way.** Check every second in that window one by one. Up to 2,880 steps. Double the length of the game and you double the work.

**With these structures.** About 12 steps. Not 12 percent fewer. Twelve total.

Both structures store pre combined summaries of chunks of the game. Any window can be built from a small number of those chunks. Each step roughly halves what is left, the same way looking up a word in a dictionary by splitting it in half beats reading from page one. That is what logarithmic time means. The work grows by one step each time the data doubles.

| Operation | Steps for a full game |
|---|---|
| take in one event | about 12 |
| points in any window | about 12 |
| biggest run in any window | about 12 |
| current score | 1 |
| the obvious approach for comparison | up to 2,880 |

## Why two structures

This is the part I would most want to explain in person.

### The Fenwick tree and why subtraction works

A Fenwick tree answers how many points were scored between minute 18 and minute 24. It stores running totals from the start of the game and then **subtracts**.

points in [18, 24) = total through 24 − total through 18

Clean and fast. It only works because of one property of addition. **It can be undone.** The first 18 minutes are counted in both totals so subtracting cancels them out exactly. Any operation that can be reversed like this works in a Fenwick tree.

### The segment tree and why subtraction breaks

Now ask a different question. What was the biggest scoring run inside minutes 18 to 24.

Try the same trick. You know the biggest run in the first 18 minutes and the biggest run in the first 24 minutes. Subtract them.

**That means nothing.** Say the best run of the whole game started at minute 17 and ended at minute 19. It sits across the boundary. Part of it is inside your window and part of it is outside. No amount of subtracting gets you the piece you want. Biggest run cannot be undone the way addition can.

So a segment tree does not subtract. It **combines**. It stores answers for small chunks of the game and merges them upward. Take the answer for minutes 18 to 21 and the answer for 21 to 24, then combine them and also check whether a better run crosses the join between them.

For that merge to work each chunk has to remember four numbers instead of one. Its total, its best run starting at its left edge, its best run ending at its right edge, and its best run anywhere inside. The edge values are what let you spot a run that spans two chunks. Four is the minimum. Drop any one of them and the merge stops working.

**The thing I would want to say out loud.** The choice between these structures is not about speed. It is about whether your operation can be reversed. Addition can, so Fenwick works and is simpler. Biggest run cannot, so it needs a segment tree.

There is a test in the repo called `test_segment_tree_prefix_subtraction_would_be_wrong` that shows the failure on a real array. I wanted this written as runnable proof instead of a claim in a comment.

## Design decisions

### One timeline instead of periods and clocks

Basketball reports time as Q3 with 4:12 remaining. Two numbers, and the clock counts down. Array indices need one number that counts up.

Every event converts to a single seconds since tip off value the moment it is created. Nothing after that point does period math. Overtime is 5 minutes instead of 12 and that is handled in one function instead of being special cased everywhere. The awkward conversion happens in exactly two places. Once on the way in and once on the way out for display.

### Watermarks and deciding when it is safe to move on

Since events arrive slightly out of order the buffer holds them briefly before passing them along. A **watermark** tracks the claim that everything before this second has arrived. Once an event is older than the watermark it is safe to release.

I got that claim wrong at first and a test caught it. The watermark has to mean nothing strictly before second X will arrive. Not nothing at or before X. The difference only shows up when two events share a second, but that happens constantly in basketball. A made shot and the assist that set it up land on the same clock reading. My first version treated a second as finished the moment it saw the first event in it, so everything else at that second got dropped as late.

**The delay setting is a real tradeoff with no free answer.** Wait too little and real events arrive too late and get dropped. Wait too long and everything reaches the scoreboard slower. There is a test that pins the failing direction on purpose. A setting whose bad value is not tested is not a documented decision. It is a number someone guessed.

I also wrote the heap underneath by hand instead of using the one built into Python, since understanding it was the point.

### Possession is a guess so it should admit when it is wrong

Play by play tells you what happened. It does not tell you who has the ball. Possession has to be worked out, and anything worked out can be wrong.

The state machine handles this by writing every transition relative to **who did the thing** rather than to what it currently believes. A made basket gives the ball to the opponent is written as opponent of the scorer, read straight off the event. So if the belief has drifted the next event corrects it instead of building on the mistake.

When an event genuinely contradicts the state, like a team shooting when it should not have the ball, the engine either raises an error or logs a violation and resyncs. It never quietly ignores it. **That contradiction is the only signal you get that something upstream broke.**

This paid off twice while I was building. My game generator opened only the first quarter with a jump ball, which left three quarters with no possession set. Later it forgot that a missed final free throw is a live ball that needs a rebound. Both times the box score looked perfect. Points, rebounds and percentages were all correct. The only evidence anything was wrong was entries in the violations list. That is what convinced me the design was right.

### Everything else stays cheap

Score, box score, streaks and lead changes update with simple counters. One step per event and no scanning. That is what makes this a streaming engine instead of something that recomputes from scratch.

The live streak counter and the segment tree look similar but answer different questions. The counter knows the current unanswered run cheaply, but only about right now. The segment tree can answer biggest run inside minutes 18 to 24. Any window, any time, including the past. Both exist because both are needed.

One small bug worth noting. I first counted a lead change only when the score flipped from one team ahead to the other in a single step. But the normal path is up 2, then tied, then down 2. It passes through a tie and neither step is a direct flip. Real games would have shown almost no lead changes. Now it compares against whoever led last and ties are counted separately.

### Wiring it together

The engine has one method that takes an event and passes it to each component in a **fixed order**. Possession first since it is the only piece that can reject an event, then the score, then the two trees.

If the trees updated first and possession then rejected the event, they would be holding points for something that never counted. Components disagreeing with each other is the worst kind of bug to chase.

The reorder buffer sits at the entrance. Everything past it is guaranteed to be in order so no other component has to worry about it. One place solves the problem instead of five places assuming it is solved.

### The commentator

A small component watches the game and writes a line when something is worth noting. An 8 point run, or a big swing in win probability.

The important part is that **it gets no special access**. It reads the game through the same three public queries anyone else would use. It cannot see the raw event history or reach inside the data structures. So every number it prints is one the engine can produce on its own, and there is a test that checks exactly that.

Win probability is a simple formula rather than a trained model. It only needs to notice a swing, not predict a winner. Adding a trained model would have been a different project bolted onto this one.

### Testing it honestly

The main technique is **differential testing**. Run the fast structure next to a deliberately dumb version that is obviously correct, on random data, and check they agree. For the heap the dumb version sorts a list and takes the first item. For the segment tree it is a plain loop over the window. Slow but clearly right.

I also lean on **conservation checks**. Things that always have to add up. Events released plus events dropped equals events received. Team totals equal the sum of the individual players. Separate time windows sum to the final score. These catch bugs that individual tests walk right past. A buffer quietly eating events would pass every ordering test and still be broken.

The two structures also check each other. One test asks the segment tree for the biggest run's time window, then asks the Fenwick tree for the score over exactly that window. Different algorithms, so agreement is real evidence instead of one component confirming itself.

The game generator produces a full realistic game from a seed. No network and no downloaded data. Same seed, same game, every time. It builds the game possession by possession instead of picking random events, so the sequences it produces are legal basketball. If it produced nonsense then every test built on it would be quietly testing the error path instead of the real one.

## What went wrong along the way

Four bugs the tests caught. I kept them here because how a bug surfaces says more than a clean history would.

1. **Watermark ties.** Events sharing a clock second were being dropped. Fixed by changing what counts as a finished second.
2. **Run windows.** The biggest run came back with the right number of points but a uselessly wide time window, because ties between equally good windows picked the leftmost one. On a mostly empty timeline a run can be padded with silent seconds without changing its total. Both randomized tests passed through this. They checked the number, not whether the answer was meaningful.
3. **Missing period openers.** Three of four quarters had no possession set. The box score was perfect. Only the violations list showed it.
4. **Missed final free throw.** Treated as a dead ball instead of a live one. Found the same way.

Numbers 3 and 4 are the ones I would point to. The output looked completely correct both times. The only reason I found them is that the state machine was built to complain instead of quietly guessing.

## Layout
src/bball/
events.py event definition, one timeline
event_queue.py heap and reordering buffer
fenwick.py Fenwick tree, points in a window
segment_tree.py segment tree, biggest run in a window
possession_fsm.py who has the ball
game_state.py box scores, streaks, leads
engine.py ties it all together
synthetic.py game generator
agent.py commentator
cli.py command line
tests/
test_structures.py data structures against brute force
test_engine.py game logic, replay, commentator, CLI

## Try it

```bash
bball replay --synthetic                      # full game with box score
bball replay --synthetic --verbose            # every event as processed
bball replay --synthetic --jitter 6           # scrambled feed, same result
bball query --synthetic --start-minute 18 --end-minute 24
```

The jitter run is the one to look at. It scrambles the incoming feed on purpose and the final box score comes out identical to the clean run. That is the reordering buffer doing its job.

The query command is the headline. One time window and both structures. The Fenwick tree answers points scored by subtracting and the segment tree answers biggest run by merging, because only one of those operations can be reversed.

## Not included

No live NBA data connection, no web API, no machine learning model. Those could be added on top of a working core. They are not pieces removed to make this look smaller.