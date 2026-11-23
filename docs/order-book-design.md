# Order-book ecology: design

Status: **approved** (2026-11-22). Stages 1 (`arena/events.py`) and 2 (`arena/book.py`) passed their
gates the same day. **Stage 3 (`arena/ecology.py`) failed twice, and the audit that followed found the
second verdict overstated and the research model broken: see "Audit of stages 1-3". Both must be
settled before any third gate or stage 4.** Stages 4-5 are not built.

Points are marked **[decided]** or **[proposed]** (a default that approval accepts). Anything
unmarked is proposed.

## Why

The arena asks whether a trading edge survives a population that adapts to it. Today everyone trades
in one batch against an automated market maker, with the same budget, and the only food is 50
identical punters. An answer from that market may not carry over to Polymarket or Kalshi, where
traders differ in motive, information, speed and money, and trade with each other through a book.

## Goals

1. A continuous limit order book with the trader roles of a real prediction market. [decided]
2. Keep the evolving ecology: strategies compete, boom, bust and take over. [decided]
3. Head-counts and capital roughly in real-life proportion at the start; evolution and growth may
   then move them. [decided]
4. See what free capital growth does before adding any limit to it. [decided]
5. Re-ask the Phase 4 questions (learned edge, hunting, decay) and see which findings survive.

**Non-goals:** exchange engineering beyond price-time priority; linked markets and arbitrage;
downloading real trade data (except the two licensed files that set retail, approved 2026-11-22);
an LLM trader; the animation (Phase 6).

**This fails if:** the mix is tuned until a hoped-for result appears; market makers live on a
subsidy; a limit is added quietly when free growth misbehaves; or old results stop reproducing.

## Approaches

| approach | for | against |
|---|---|---|
| **Beside the current market [decided]** | old results reproduce; every experiment can run on both | two markets until this one is proven |
| Replace the current market | less code | old results lost; can't tell the book's effect from the roles' |
| Add roles to the current market first | smallest step | market makers and speed mean nothing in a batch auction |

## The event

- The lead does a random walk with news jumps and settles at 1 or 0, as today. [decided]
- The true probability uses the exact formula, fixing the README's known issue: [decided]
  `p = sum over k of Binom(k; left, 0.02) * Phi(lead / sqrt(left + 25 * k))`.
- Starting leads are spread (s.d. 1 x sqrt(300)). Kalshi's "two-thirds of prices under 10c or over
  90c" counts the last trade of each day, mostly at the close, so it can't be compared with all
  ticks. Sampled the paper's way this spread gives 57-62% (paper 68%) and 2% of markets open beyond
  1-99c. A spread of 2 overshoots (74-77%) and opens 24% of markets already decided. [proposed]
- Each market opens with a public reference price: the exact chance given one public, noisy look at
  the starting lead (0.3 of log-odds off at 10-90c). Without it makers would quote 50c on a 97c event
  and be picked off on tick 1. [decided: research design]
- Prices can't go beyond 1-99c, so accuracy is measured against the truth clipped to that range.

## The market

- Whole cents, whole contracts, price then time priority, full collateral when an order is posted.
- Each trader has one bid and one offer, replaced when it next arrives, and can't trade with itself.
- YES and NO are mirror images (settled in stage 2): a trader whose bid is at or above its own offer
  posts neither, and whichever of its sides trades at once is placed first, since what that frees
  backs the other side. Dropping one side, or always placing the bid first, would favour one outcome.
- Budgets are per trader. The only position limit is collateral.
- Arrival order is random, weighted by the role's speed: makers and news traders are usually first,
  retail and institutions last. [proposed]
- Each fill records who was maker and who was taker.
- 400 markets a generation run side by side. A professional's capital is split equally between the
  markets it enters, and it enters fewer if that would be under $5 a market. [proposed]
- Positions are public after each tick, as on chain.
- No fees at first. Fees are a later sensitivity run (Kalshi: 0.07 x P x (1 - P) a contract, paid by
  takers, in the paper's period).
- Rules that need a price use the last tick's average trade price (the opening reference until the
  first trade). Accuracy is measured on the midpoint when the book is two-sided, since trade prices
  bounce between bid and ask, and on the last trade otherwise.

## The roles

| role | accounts at start | capital at start | share of money | who sets it |
|---|---|---|---|---|
| retail | 2,750 | lognormal, median $100, log s.d. 2 | 45% | fixed, topped up each generation |
| hedge funds | 90 | lognormal, median $10k, log s.d. 1 | 33% | evolves |
| market makers | 10 | $25k each | 6% | evolves |
| institutions | 3 | $250k each | 17% | fixed, topped up |
| insiders | 3 | $5k each | 0.3% | fixed; off except in stress tests |

The fund, maker and insider head-counts follow published Polymarket figures; institutions' is an
assumption. **The capital figures are assumptions** and get sensitivity runs at 3x higher and lower.
The retail row is the approved start; since gate 2 retail's money is one turnover of its orders
(about $1.4M at that setting). Share of money is not share of trading. The 100
professional slots are fixed, and funds and makers compete for them. [proposed]

**Retail.** A stream of small orders, not 2,750 bots, each order in its own slot. [proposed]

- Measured on July 2024 Polymarket fills by `arena/retail_data.py`: retail is wallets under $10,000
  in the month, mostly takers, not machine-like. [decided: real data]
- Sizes are in contracts: lognormal, median 20 and mean 205.3.
- Side is a coin flip. In contracts that pushes neither way; a coin flip on dollars would buy 16 net
  YES contracts per $1 at 3c and build in a favourite-longshot bias by accident.
- Every account trades at least once, then in proportion to its money, and never spends more than it
  has.
- An order won't pay more than 0.759 of log-odds past the price (18c at 50c), rounded to the same
  number of cents both ways and at least 1c. What doesn't fill is cancelled. **This limit is an
  assumption:** 0.759 is the 90th percentile distance of retail's fills from the last price, spread
  and price moves included, not what retail was willing to pay.
- The number of orders is set at stage 3 so that retail fills 16.6% of 9,000 contracts (its measured
  share of the median market) in the median market. Retail's money is one turnover of its orders.
  Total volume is a check, not a target.
- No longshot bias of its own in the main run, so any that appears is a result. One extra run
  switches a bias on. [decided]

**Institutions.** Each works 25% of its capital a generation across 20 random markets, in a random
direction, spread evenly over the ticks with a price limit, measured from the price when the order
arrived (never the last trade, which is its own: see "Stage 3 result"): about $3,000 a market, and about 10% as
much money as retail trades overall. 5% and 50% are sensitivity runs. They trade for outside
reasons. [decided] (Within the price limit they fill only 4-7% of capital: see the audit.)

**Insiders.** When switched on, each sees the truth 10 ticks ahead in 4 markets. Trades can't be
scored at the truth in those runs (it is no longer fair given who traded), so they use real profit.
[proposed]

**Hedge funds** keep their sub-strategies [decided]; 15 of each at the start:

| strategy | sees | genes besides size |
|---|---|---|
| momentum | past prices | lookback, price limit |
| contrarian | past prices | lookback, price limit |
| forecaster | its estimate, 2 ticks late | research spend, order aggressiveness |
| news | a jump's change in log-odds, with error set by its research, immediately; not the level | research spend, price limit |
| copier | public positions, leaderboard | spot or locked wallet, board, lag, price limit |
| fader | public positions, leaderboard | spot or locked wallet, board, lag, price limit |

Size is always a fraction of capital, so it means the same for a small and a large fund; a copier
copies the watched trader's position as a fraction of that trader's capital. Copiers
can only watch professionals, institutions and insiders; the leaderboards are profit and return on
capital. [proposed]

**Market makers.** Genes: half-spread, quote size as a fraction of capital, inventory shading, response
to order flow, news reaction. [proposed]

- **Centre:** its own estimate of the price, starting at the opening reference, shaded by inventory
  when quoting. [decided]
  - After each tick it moves by `response x f` cents and stays within 1-99c, where f is what customers
    (everyone but makers) bought minus what they sold, whether they took or a maker crossed to their
    resting order (trades between makers don't count; between customers, the taker's direction does), divided by what all makers had
    quoted on that side and clipped to -1..1 (if they had quoted nothing, f is just the sign).
  - It never reads the midpoint or trade prices: both contain the makers' own shaded quotes. In a
    one-maker test with uninformed traders only, centring on the last trade wandered 30-42c from a
    fixed true value with shading on; order flow wandered 3-13c at responses of 1-3c.
  - A slow response is late to news and a fast one is pushed around by noise, which is the gene's
    trade-off.
- **Edges:** near 1c and 99c the half-spread narrows equally on both sides. Cutting off only one
  side (1c/7c around a centre of 3c) would overprice longshots by itself. [proposed]
- **News:** makers see that a news event has happened, but not its direction or size. The news gene
  sets how far they widen their quotes when it does, up to pulling them altogether, and for how
  many ticks. [decided]

## Accuracy

Accuracy evolves with a random component and can get worse as well as better. [decided]

- `spend` is inherited, mutates both ways at birth and is then fixed for life. It costs that many
  dollars per market entered and can't exceed the bot's cash.
- It buys noisy looks at the lead, and the estimate is the best guess from them
  ([research-error-design.md](research-error-design.md)). [decided] A look's noise is
  `3.73 x (0.25 / spend)` lead points, so 4x the spend halves the error, and $0.25 gives an average
  error of 0.057, as the old fixed error of 0.29 log-odds did.
- A bot's look noise is that times `exp(2z)` (its error about `exp(z)`), with z drawn at birth
  (s.d. 0.2). Luck isn't inherited.
- Each generation `z = rho * z + noise`, keeping s.d. 0.2, with rho 0.9 (half of any luck gone in 7
  generations); 0 and 1 are sensitivity runs. It can't be set from the published "44% stay skilled":
  that study finds 3% of accounts skilled, our test finds 27-35% of forecasters, and here 44% is
  reached only with luck redrawn every generation (43%; 100% with fixed luck). [proposed]
- Half of each look's noise variance is shared by everyone in a market.

## Evolution and capital

- Strategy, accuracy and size evolve. [decided]
- Capital grows freely: profit compounds, with no cap per fund or role. [decided]
- A control run trades with reset capital on the same markets, so that the effect of growth can be
  seen. It tracks what capital would have been, so exits and breeding follow the same rules. [proposed]
- Who leaves: [decided]
  - any professional down to half of its peak capital;
  - any that has traded under 1% of its capital in each of the last 5 generations;
  - of those at least 5 generations old, up to the 10 with the worst return on capital over the
    last 5, and only if that return is negative.
- Nobody retires, and a profitable fund never leaves for being big.
- The slot goes to a child of a tournament winner (best of 3 on return on capital after
  research, summed over the last 5 generations, counting generations before birth as 0) with fresh
  outside capital drawn for its role. 5% of children switch strategy and then start at
  that strategy's usual settings.
- Every gene has its own mutation step, in proportion to its value plus a small absolute part.
- A ledger records money in (starting capital, top-ups, births) and out (exits, research, what retail,
  institutions and insiders take away when they are reset) and must balance to the cent.

## Tuned, and checked

Set from published figures: starting-price spread, head-counts, and a market's size. Set from
Polymarket fills: retail's share, order sizes and price tolerance.
Kalshi's figures come from contracts with over $1,000 of volume, open at least a day; the
head-counts are Polymarket's.

Never tuned to, only compared afterwards:

- makers do better than takers. Direction only: Kalshi's -9.6% and -31.5% are averages per contract,
  with fees for takers and a strong longshot bias;
- which way any favourite-longshot bias points (Kalshi: longshots overpriced).

Reported, but not like for like, since the studies cover years and this covers one generation:

- the share of accounts that lose (67% there; a coin-flip retail account loses about 54% of the time);
- the top 1% of winners' share of profit (76% there; the assumed spread of capital drives part of it).

## Stages, each with a gate

| stage | build | go on only if |
|---|---|---|
| 1 | `arena/events.py` | outcomes match p within sampling error at 100, 10 and 1 ticks left, starts included |
| 2 | `arena/book.py`, from the prototype | it matches the brute-force reference, money balances, under 5x slower |
| 3 | `arena/ecology.py`, population frozen | both maker checks below pass, still under 5x slower with every strategy running, book two-sided on over 80% of ticks, gap under half of what it is without informed traders, and some maker setting pays for itself on fresh seeds, also where no institution trades |
| 4 | evolution, free growth and reset control | 3 seeds x 1000 generations finish with the ledger balanced, whatever they show |
| 5 | learning agent, hunting, decay on the book | results reported beside the old market's |

- Maker checks: with no trades, no news and no change in inventory a maker's quotes don't move, and
  shading must not compound through trades; on news they widen or vanish, by the same amount
  whichever way the news points.
- At stage 3 every gene except the makers' is written down before the first run. Only maker
  settings, rho's sensitivity and retail's order count are varied.
- If stage 3 fails, stop and revisit the design. Don't subsidise makers.
- Free growth is reported, not repaired: top fund's share of capital, inequality, each role's share,
  extinctions and gap. Any limit is a separate decision afterwards.
- Expected-profit scoring must match real profit within sampling error before stage 5.
- The learning agent sees what it sees today plus spread and depth, has a median fund's capital, and
  places a limit order at its shifted valuation.
- `arena/market.py` and the Phase 1-4 code paths keep their current behaviour. numba joined
  `requirements.txt` at stage 2. `prototypes/` is deleted once stage 3 passes.

## Stage 3 result: failed

The four numeric gates passed on fresh seeds (0.47x speed, book two-sided 97%, gap 0.093 against
0.204, makers +50 +- 12 a market). The pass was false: the makers lived on a subsidy, which this
design counts as failure. The gate now tests for that too.

- **Institutions chased their own trades.** Their limit is measured from the last trade, which is
  their own, so they walked the price 46c past the truth and paid 88.7c for what was worth 55.5c.
  They supplied 97% of everyone's losses. In the 85% of markets without one, and with them switched
  off, makers lost about 120 a market in 10 of 10 seeds.
- **Informed traders take far more from makers than retail gives:** 5-6x at 90 orders a market,
  almost 60x at the calibrated 6.9.
- **Retail barely trades:** 25% of its contracts fill (the rest are past its 0.2 price limit or find
  too little size or money), and at the calibrated order count it uses under 2% of its money.
- **The volume target couldn't calibrate retail:** informed and maker trading moves volume more than
  retail does, so the search ended at the fewest orders allowed.

Revisions tried on diagnostic seeds only, as patched copies (none adopted; a small search each):

| revision | makers a market | gap (without informed) | median volume |
|---|---|---|---|
| none, institutions off | -42 | 0.071 | 3,248 |
| A: institution limit from its arrival price. B: makers learn from every customer trade. Retail 2x | -34 | 0.070 (0.202) | 4,888 |
| A + C: retail accepts 0.5 of log-odds past the price. Retail 2x | +40 | 0.118 (0.180) | 10,194 |
| A + B + C | -9 | 0.066 | 4,251 |
| A + B + C, retail 1.5x | +13 (+0 without institutions) | 0.089 (0.203) | 7,516 |
| A + B + C, retail 2x | +45 (+31 without institutions) | 0.095 (0.204) | 8,922 |
| A + B + C, retail 3x | +126 | 0.123 (0.184) | 13,549 |

A and B correct errors. C and retail's money decide whether makers survive, and there is no data
here for either.

**Since then** (as written before the second gate; corrected by the audit below):

- A and B are fixed in `arena/ecology.py`, with a check for each. On diagnostic seeds institutions
  now lose 25 a market, and no other role was found walking the price. Makers still lose about 120 a
  market at the old settings, as expected until retail is set from evidence. The gate hasn't been
  re-run; it will need a new search, rules written down first and unused seeds.
- Kalshi's API can't be that evidence. Its Developer Agreement (v1.1, section 3) limits the API to a
  member's own trading and bans collecting or storing the data for anything else. Nothing was
  downloaded from it.
- Candidates, not yet approved or downloaded, both CC BY 4.0: one month of Polymarket on-chain fills
  with wallets (Qin and Yang, `TimeSeventeen/Polymarket-v1`, `OrderFilled/2024_07.parquet`, 34 MB),
  and daily taker costs by order size (Zenodo 24292871, `realized-cost-daily.csv`, 1 MB). Neither
  shows orders that didn't fill, so a price tolerance measured from them was expected to be a lower bound (it isn't: see the audit).

## Stage 3, second gate: failed, at a badly chosen setting

Retail was measured with definitions written down before the data was downloaded. Rules A and B were
fixed. One gate was run, on seeds never used before. Standard errors are by seed (first reported by
market, as 2.2 and 2.1, which was too small: a seed's markets share their population).

| test | result |
|---|---|
| under 5x slower | 0.69x, pass |
| two-sided on over 80% of ticks | 99.4%, pass |
| gap under half of that without informed traders | 0.075 vs 0.186, pass |
| makers pay for themselves | -32.3 +- 2.8 a market, **fail** |
| ...also where no institution trades | -48.7 +- 2.7 a market, **fail** |

**It doesn't show that makers can't survive.** The search ranked settings at one order count, where
they fill different amounts of retail. The winner was picked with retail 55% over its target and
then judged at the target. Re-checked on other seeds, each setting at equal retail:

| blind makers | makers a market | where no institution trades |
|---|---|---|
| the setting it chose (half-spread 10c, size 0.5) | -37.2 +- 4.6 | -53.9 |
| its runner-up (10c, size 0.2) | -2.4 +- 2.8 | -13.2 |
| the chosen one, leaving for the last 35 ticks | +3.1 +- 1.4 | -6.2 |
| outside the grid (15c, size 0.2, shading 50) | +24.2 +- 1.5 | +14.7 |

The last clears both profit tests on exploratory seeds, but it is a cartel: the book's median spread
is 10.8c, and one maker cutting to 10c more than doubles its own profit (+2.4 to +5.4).

Prices are calibrated (0.007), but not healthy throughout: the favourite-longshot slope is 1.10
(prices too timid; the cheap-price measure finds no bias, -0.0009), 5% of news is priced in after a
tick and 82% after 20. The checks that were never tuned to: makers did not beat takers (+5.9% vs
+6.4% a contract), 41% of retail accounts lose (67% there) and the top 1% take 50% of profit (76%
there).

The measurement's cleaning reproduces an independent study's order counts and walk inside orders in
all 8 size and price groups. That validates the cleaning, not the wallet rule, share or tolerance.

The sealed sensitivities, run after the verdict. They are exploratory: 20 settings, the same search
bias, 4 seeds, retail up to 9% off its target, and no warm-up generation, so copiers and faders were
idle (the gate's setting scores -25 there against -32).

| retail defined or measured as | makers a market | where no institution trades |
|---|---|---|
| under $1,000 | -26 | -26 |
| under $10,000, 90th percentile distance (the gate) | -25 | -40 |
| under $100,000 | +8 (not significant) | -8 |
| 75th percentile distance | -52 | -62 |
| 95th percentile distance | +23 | +14 |

Candidate changes from the same flawed diagnostic, none adopted: informed funds half as big -32;
forecasters more passive -56; no news traders -2 (with other maker genes, so it shows no cause);
5 forecasters +23; makers with research +62; half research, half order flow +50. The research
figures are withdrawn (below).

## Audit of stages 1-3 (2026-11-22)

The code, sealed rules, logs and data were re-checked, with findings reproduced on seeds kept
apart from any gate's.

**Held up:** the exact probability, matching, money, no look-ahead, the retail data's cleaning and
arithmetic, gate 2's numbers to the cent, and the old market's reference results.

**Fixed, with checks:**

- A fund that bought no research had perfect information for free, and cheap research rounded to free.
  Now no research means no trading, and costs round up.
- The search compares every setting at the same retail volume, and warns at the edge of its grid.
- Maker profit is after research, with errors by seed.

**Open. No third gate before these are decided:**

1. **Research is close to meaningless, which is fatal to maker research.** A trader's error is fixed
   for the whole market, so its estimate moves exactly with the truth, whatever it paid. On narrow
   quotes (half-spread 3c, size 0.1): paid research +39.4 +- 2.1 a market, the public opening price plus the truth's
   moves for free +41.5 +- 2.2, research 10x cheaper +32.0, 10x dearer +18.0. Accuracy can't evolve
   meaningfully at stage 4 either. The current market has the same flaw (README, Known issues).
   **Fixed in the order book** by [research-error-design.md](research-error-design.md) (step 1): at gate 2's
   setting on 6 new seeds makers go from -34.0 to -16.9 a market. Not yet in the current market.
2. **Retail's numbers don't mean what they were called.**
   - The 0.759 isn't what retail will pay. The median distance is 1c and the walk inside an order
     0.2c; the top tenth is mostly the spread and price moves since the last trade.
   - Filled sizes became intended sizes. 22% of intended contracts fill, mean 53 a filled order
     against 205, as the book is shallow and accounts run out of money.
   - Retail's loss against the last trade is 5.0c a contract against 2.8c (1.0-4.7) in the data, but
     3.4c against 3.1c (2.2-4.1) with markets equal: the excess is in large orders. The walk is 1.6c
     against 0.2c.
   - 1,498 contracts is a Polymarket share of a Kalshi volume. Polymarket's median market had 2,887
     of retail in the month (16,404 in all).
3. **Identical makers searched as a group act as a cartel.** A gate needs a limit on the spread and
   no profitable undercut.
4. **Makers can't act on the time left**, though it is public, and that's where they lose.
5. **Smaller:** institutions fill 4-7% of capital, not the 25% they aim at; log-odds stop at +-6, so
   news traders see 41% of news in the last 50 ticks as no move; holding the opening price (gap
   0.175) beats the blind benchmark (0.186), so that gate is weak. The capital, institution-share,
   fee and longshot sensitivities haven't been run. Rho's has: makers -34.4, -33.5 and -33.4 at
   1, 0.9 and 0.

## Risks

- **Depth.** All makers together hold about $625 a market: 1,250 contracts of one-way flow at 50c
  and under 700 on the expensive side near 0 or 1, against retail orders meant to average 205. The
  prototype's "makers need 13% retail activity" was measured at a much smaller scale and doesn't
  carry over; stage 3 measures it again.
- **Runaway funds.** A large fund may become the market. That is a finding.
- **Evolution may stall.** If nobody loses there are no births. Births a generation are reported
  from stage 4's first run.
- **Research gets cheaper with size.** Its cost is in dollars, so under free growth a big fund can
  afford better research. Luck and drift still apply; the reset control shows the effect.
- **Labels.** Real data doesn't say who is a hedge fund. "3.14% skilled" counts winners, so 90
  funds is only a starting point.
- **Evidence for the maker rule** is a one-maker toy model. Stage 3 is the real test.
- **Equal markets.** Real volume is very uneven between markets; here every market is the same size.
- **Speed.** Stage 2 measured the book's own work at 246 slots as 0.6-1.3x a current generation at the
  expected load and up to about 2.3x at the heaviest, allowing for how much the current market's
  timing varies. With every strategy running stage 3 measured 0.69x.

## Left to the builder

Data layout, mutation step values (stated and tested for heritability), starting gene ranges, the
speed weights, plotting, and how stage 3 searches maker settings.

## Sources

- Skilled accounts, persistence, losers (secondary report of Gomez-Cram, Guo, Jensen and Kung):
  [The Block](https://www.theblock.co/amp/post/398902/skilled-polymarket-traders-are-a-3-minority-and-everyone-else-funds-their-gains-study)
- Profit concentration: [Akey, Gregoire, Harvie and Martineau](https://ideas.repec.org/p/cpr/ceprdp/21615.html)
- Kalshi prices, stakes, volumes, makers and takers:
  [Buergi, Deng and Whelan](https://www2.gwu.edu/~forcpgm/2026-001.pdf)
- Polymarket trade sizes (secondary report):
  [Casino.org](https://www.casino.org/news/prediction-market-users-mostly-small-act-like-day-traders/)
