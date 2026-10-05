import type { ReactNode } from "react";

/* Long-form explainer: what DML is, from the estimand down to the inference, with the
   project numbers wired in so every concept maps to something visible in the Playground
   tab. Written to stand alone for someone who has never seen a residual. */

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section className="card prose" id={id}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Analogy({ children }: { children: ReactNode }) {
  return <div className="callout analogy"><strong>Analogy.</strong> {children}</div>;
}

function Example({ title, children }: { title: string; children: ReactNode }) {
  return <div className="callout example"><strong>Worked example — {title}.</strong> {children}</div>;
}

function Warning({ children }: { children: ReactNode }) {
  return <div className="callout warning"><strong>Watch out.</strong> {children}</div>;
}

const F = ({ children }: { children: ReactNode }) => <span className="formula">{children}</span>;

const WORKED = [
  { m: "A", x: "falling", s: -2.0, y: 0.5 },
  { m: "B", x: "falling", s: 0.0, y: 2.0 },
  { m: "C", x: "rising",  s: 1.0, y: -1.5 },
  { m: "D", x: "rising",  s: 3.0, y: -0.5 },
];

export default function Explain() {
  return (
    <div className="explain">
      <div className="card prose tldr">
        <h2>What is DML, in one paragraph</h2>
        <p>
          <strong>Double machine learning</strong> is a way to estimate <em>one</em> number — the causal effect of a
          treatment on an outcome — when you have a lot of other variables you'd like to control for, too many to
          handle with plain regression. The trick: use machine learning to predict the outcome and the treatment
          from the controls, <em>on data the model hasn't seen</em>; throw the predictions away and keep only the
          prediction <em>errors</em>; then estimate the effect from those errors with a simple closed-form formula.
          Because the final step is a simple equation rather than a black box, you get honest standard errors and
          confidence intervals — and because the ML part is only asked to make predictions (not to answer the causal
          question), its mistakes contaminate the answer far less than intuition suggests. In this app the treatment
          is a Fed policy surprise, the outcome is the equity move around the announcement, and the controls are
          things like the level of 1-year yields and recent surprise history.
        </p>
      </div>

      <nav className="card toc">
        <strong>Contents</strong>
        <ol>
          <li><a href="#estimand">The question we're actually answering</a></li>
          <li><a href="#naive">Why the obvious regressions fail</a></li>
          <li><a href="#plr">The partially linear model</a></li>
          <li><a href="#fwl">Partialling out: the engine of the whole method</a></li>
          <li><a href="#overfit">Why residualizing once isn't enough: overfitting bias</a></li>
          <li><a href="#crossfit">Cross-fitting: grading on a held-out exam</a></li>
          <li><a href="#moment">The estimating equation and why it's robust</a></li>
          <li><a href="#nuisance">What "good nuisances" look like</a></li>
          <li><a href="#inference">Getting honest error bars</a></li>
          <li><a href="#ui">Every knob in the Playground, mapped to the math</a></li>
          <li><a href="#diagnostics">Diagnostics and failure modes</a></li>
        </ol>
      </nav>

      <Section id="estimand" title="1 · The question we're actually answering">
        <p>
          Q1 asks: <strong>how much do equities move, in percent, per 10 basis points of Fed surprise?</strong> A
          "surprise" is the part of the announcement markets didn't expect — measured here by the change in
          short-rate futures in a tight window around the statement. The answer is a single slope, θ. If θ = −1, a
          statement that was 10bp more hawkish than expected came with a −1% equity move, on average, holding
          everything else fixed.
        </p>
        <p>
          "Holding everything else fixed" is doing all the work. We never observe two worlds, one with the surprise
          and one without. We observe a history of meetings, each with its own market backdrop, and we have to
          reconstruct the counterfactual from that. That reconstruction is what the whole machinery is for.
        </p>
        <Analogy>
          You want to know whether a study technique improves exam scores. You can't clone students. But you can
          compare students who used the technique to students who didn't — provided the two groups would have scored
          similarly without it. DML is the machinery for making that "provided" quantitative when "would have
          scored similarly" depends on dozens of background variables (GPA, sleep, prior courses...).
        </Analogy>
        <p>
          The reason this is hard specifically for Fed surprises: <strong>the Fed and the market react to each other</strong>.
          Bad inflation data moves both yields and equities; the Fed's statement responds to the same data. So the
          raw co-movement of surprises and equity returns mixes at least two channels: the causal effect we want, and
          a common-reflex channel where both series respond to the economic news of the day. Controlling for the
          state of the world (X) is how we shut the second channel.
        </p>
      </Section>

      <Section id="naive" title="2 · Why the obvious regressions fail">
        <p>Two obvious strategies, two opposite failures:</p>
        <h3>Too few controls: omitted variable bias</h3>
        <p>
          Regress equity moves on surprises alone. If hawkish surprises tend to arrive during fragile, falling
          markets (they do), the regression credits the surprise with damage actually caused by the backdrop. The
          estimate is a cocktail of the true effect and the confounding.
        </p>
        <Analogy>
          Estimating the effect of a fitness program by comparing members' health to non-members, ignoring that
          members were healthier to begin with. You'd credit the program with pre-existing differences.
        </Analogy>
        <h3>Too many controls: overfitting</h3>
        <p>
          So add controls — but now try adding dozens or hundreds: every yield, every spread, every lag, every
          interaction. Plain regression falls apart three ways. <em>Statistical</em>: each control eats a degree of
          freedom, the estimates get noisy, and correlated controls make the coefficients numerically unstable
          (multicollinearity). <em>Computational</em>: you can't include nonlinear terms (say, "VIX matters only when
          it's above 20") without manually inventing them. <em>Conceptual</em>: with more controls than meetings —
          we have a few hundred meetings — the regression has more knobs than observations and fits noise perfectly.
        </p>
        <Analogy>
          Retouching a photo: too few sliders and the blemish stays; a thousand sliders and you end up painting over
          the face — every pore is "corrected", and the person in the photo is gone. Overfit regression doesn't just
          fail to remove confounding, it manufactures fake precision.
        </Analogy>
        <p>
          DML threads the needle: it lets you throw arbitrarily rich, flexible controls at the problem through
          machine learning, but estimates θ itself with a simple two-variable formula that you can write down and
          audit. ML handles the thousand-slider part; the closed-form part stays transparent.
        </p>
      </Section>

      <Section id="plr" title="3 · The partially linear model">
        <p>DML needs you to commit to a shape for the problem. The one used throughout this app is the{" "}
          <strong>partially linear model</strong>:</p>
        <F>Y<sub>t</sub> = θ·s<sub>t</sub> + g(X<sub>t</sub>) + ε<sub>t</sub></F>
        <p>Read it left to right, term by term:</p>
        <ul>
          <li><F>Y<sub>t</sub></F> — the outcome: the equity move around meeting <em>t</em>, in percent.</li>
          <li><F>s<sub>t</sub></F> — the treatment: the surprise, in units of 10bp of 1-year-yield equivalent. θ is
            the object of interest: <strong>percent equity response per 10bp surprise</strong>.</li>
          <li><F>g(X<sub>t</sub>)</F> — a <em>nuisance function</em>: the part of the equity move explained by the
            pre-announcement state of the world (yields, recent surprise history, VIX...). "Nuisance" is the technical
            term — we don't care about g itself; it exists to be removed. It can be <em>any</em> function: linear,
            wiggly, with interactions. We never have to write it down.</li>
          <li><F>ε<sub>t</sub></F> — what's left: the equity move not explained by the surprise or the backdrop —
            idiosyncratic news, noise. Assumed, as always, to be uncorrelated with the surprise once the backdrop is
            accounted for.</li>
        </ul>
        <p>
          Two commitments are hidden in this equation. First, the treatment enters <em>linearly</em> and with a
          constant slope: a 10bp surprise moves equities by θ whether it's 2005 or 2020, whether the surprise is up
          or down. (Allowing θ to vary by context is a different question — that's Q3, heterogeneity. Not here.)
          Second, everything confounding goes through X <em>before</em> the announcement: the controls must be
          pre-treatment. Using a same-day variable that itself reacts to the statement would be controlling away
          part of the effect. This is why the app's control set is built only from pre-event data — the 1y yield is
          read from the last close <em>strictly before</em> the announcement, for example.
        </p>
      </Section>

      <Section id="fwl" title="4 · Partialling out: the engine of the whole method">
        <p>
          Suppose for a moment we <em>knew</em> the two conditional expectations{" "}
          <F>l(X) = E[Y|X]</F> and <F>m(X) = E[s|X]</F>. Then we could compute, for every meeting, two residuals:
        </p>
        <F>ỹ = Y − l(X) &nbsp;&nbsp;·&nbsp;&nbsp; s̃ = s − m(X)</F>
        <p>
          ỹ is the part of the equity move the backdrop <em>couldn't</em> explain; s̃ is the part of the surprise the
          backdrop <em>couldn't</em> explain — the genuinely unexpected component. A 70-year-old theorem (Frisch–Waugh–Lovell,
          later generalized to this form by Robinson, 1988) says: <strong>θ is exactly the slope of a regression of ỹ on
          s̃ through the origin</strong>:
        </p>
        <F>θ = Σ<sub>t</sub> s̃<sub>t</sub>·ỹ<sub>t</sub> &nbsp;/&nbsp; Σ<sub>t</sub> s̃<sub>t</sub>²</F>
        <p>
          Intuition: after subtracting what was predictable from both series, whatever co-movement is left between
          the <em>unpredictable</em> parts of the surprise and the move can only come from the surprise causing the
          move — the common-reflex channel has been subtracted out of both. The denominator Σs̃² measures how much
          genuinely unpredictable surprise variation there is; θ is a precision-weighted average of the per-meeting
          slopes ỹ/s̃, weighted by how surprising each meeting was.
        </p>
        <Analogy>
          Judging a surprise party's effect on someone's mood: don't compare mood at the party to mood on ordinary
          days (confounded — parties happen on birthdays). Compare <em>deviations</em>: how much happier they are
          than you predicted for that day, versus how much more surprising the party was than typical for their
          friend group. If the party was exactly the kind their friends always throw (s̃ ≈ 0), it carries almost no
          information about parties-in-general, so it gets almost no weight.
        </Analogy>

        <Example title="four meetings, by hand">
          <p>
            Tiny dataset, two market regimes. X is binary: rates falling or rising. In rising-rate periods, surprises
            skew positive (Fed climbing) <em>and</em> equities skew negative — a confounder that correlates with both
            treatment and outcome:
          </p>
          <table className="table">
            <thead><tr><th>meeting</th><th>regime X</th><th>surprise s</th><th>equity move y</th></tr></thead>
            <tbody>
              {WORKED.map((r) => (
                <tr key={r.m}><td>{r.m}</td><td>{r.x}</td><td>{r.s.toFixed(1)}</td><td>{r.y.toFixed(1)}</td></tr>
              ))}
            </tbody>
          </table>
          <p><strong>The naive regression</strong> ignores X: θ_naive = Σs·y / Σs² = (−1 + 0 − 1.5 − 1.5)/(4+0+1+9) =
            −4/14 ≈ <strong>−0.29</strong>. Negative! Hawkish surprises <em>raise</em> equities?? That's the confounding
            talking: rising-rate meetings carry positive surprises and negative returns, and pooling smears that
            pattern onto the treatment.</p>
          <p><strong>Partialling out</strong> instead removes each regime's mean first. Falling regime: mean surprise
            −1, mean move +1.25 → residuals (s̃, ỹ) = (−1, −0.75) and (+1, +0.75). Rising regime: mean surprise +2, mean
            move −1 → residuals (−1, −0.5) and (+1, +0.5). Now the regime skew is gone from both series, and:</p>
          <F>θ = [(−1)(−0.75) + (1)(0.75) + (−1)(−0.5) + (1)(0.5)] / 4 = 2.5/4 = <strong>+0.625</strong></F>
          <p>
            Sign flipped. Within each regime, more-surprising-than-expected statements went with better-than-expected
            equity moves — the causal effect, unobscured. Note what happened mechanically: meetings that looked
            "positive surprise, negative return" were exactly the meetings where both were <em>expected given the
            regime</em>, so after subtraction they carried little signal.
          </p>
        </Example>
        <p>
          That example used group means as the "nuisance model". In the app, l̂ and m̂ come from ridge, elastic net,
          or random forests over the full control set — the same logic, with a much richer model of "what was
          expected".
        </p>
      </Section>

      <Section id="overfit" title="5 · Why residualizing once isn't enough: overfitting bias">
        <p>
          So: estimate l̂ and m̂ with ML on all the data, residualize, apply the formula? Not yet — this has a subtle,
          serious flaw. If l̂ and m̂ are fit <em>on the same meetings we then score</em>, an overflexible model can
          partially "explain away" the treatment itself.
        </p>
        <Analogy>
          Grading students on the same questions you taught from: a good tutor (flexible model) raises measured
          scores by memorizing quirks, not by teaching. Their "improvement" (residual) is contaminated by the fit.
          Worse, the bias doesn't average out with more meetings — it's built into how the model was fit.
        </Analogy>
        <p>
          Concretely: if m̂ overfits, it absorbs part of the true variation in s into its predictions, so s̃ is
          <em>smaller</em> than the true unpredictable component — the denominator Σs̃² shrinks toward zero and θ
          becomes unstable, sign-flipping, or explodes. This is <strong>regularization bias</strong>, and it doesn't
          vanish as the sample grows, because it comes from the model-fitting step, not from sampling noise. A model
          that's too rigid (ridge with huge penalty) has the opposite problem: it leaves confounding in the
          residuals. Either way, θ inherits the nuisance model's mistakes.
        </p>
        <Warning>
          The app will literally refuse to estimate when this goes too far: if m̂ explains s <em>perfectly</em>, the
          denominator is zero and <code>partially_linear_dml</code> raises "no residual treatment variation". With
          1-nearest-neighbor as the nuisance, every meeting predicts itself and that's exactly what you get. With
          subtler overfitting you get no error — just a quietly wrong answer. That's worse.
        </Warning>
      </Section>

      <Section id="crossfit" title="6 · Cross-fitting: grading on a held-out exam">
        <p>
          The fix is disarmingly simple: <strong>never let a model predict a meeting it was trained on</strong>. Split
          the timeline into folds. For each fold, fit the nuisance models on the training meetings and predict only
          the held-out meetings. Every meeting's residual comes from a model that had <em>no access to it</em> — so
          overfitting can't flatter the residuals; prediction errors behave like honest forecast errors.
        </p>
        <Analogy>
          K-fold cross-validation, which you may know from ML coursework, but with a twist: the exam questions are
          graded against the "expected given the backdrop" predictions, and the final grade (θ) is computed from the
          exam errors, not from the training scores.
        </Analogy>
        <p>
          Two project-specific choices matter here. <strong>Chronological splits</strong>: random folds would let a
          model train on 2015 and "predict" 2008 — impossible in real time and illegitimate for a time series.
          <F>walk_forward</F> uses expanding windows: each fold trains on everything before a test block and is
          tested on that block, marching forward. <strong>Discarded meetings</strong>: the first <code>min_train</code>{" "}
          meetings train the very first nuisance models and are never scored. They are the price of never scoring on
          your own training data — in the primary run that's 82 of 163 meetings. The app reports them explicitly
          (<em>n_meetings_discarded</em>) because they define the population the estimate actually describes.
        </p>
        <p>
          <code>embargo</code> drops extra meetings between train and test. It's insurance against outcome windows
          that spill over: if you ever use a t+20 outcome, the outcome of a training meeting could overlap the test
          window. Same-day outcomes (everything in Q1 today) can't reach the next meeting, so the default is 0 — but
          the knob is there for exactly that future case.
        </p>
      </Section>

      <Section id="moment" title="7 · The estimating equation and why it's robust">
        <p>
          The final θ is the solution of one equation — the <strong>moment condition</strong>:
        </p>
        <F>Σ<sub>t</sub> s̃<sub>t</sub> · (ỹ<sub>t</sub> − θ·s̃<sub>t</sub>) = 0</F>
        <p>
          i.e., the residuals of the <em>structural</em> relationship (ỹ minus what θ predicts) must be uncorrelated
          with the treatment residual s̃. Solving: θ = Σs̃ỹ / Σs̃² — the same formula as before. What makes DML
          special isn't this equation (it's old — Robinson, 1988) but a property it has at the solution, called{" "}
          <strong>Neyman orthogonality</strong>: the equation's answer is insensitive, to <em>first order</em>, to
          errors in the nuisance models.
        </p>
        <p>
          Why: θ depends on l̂ and m̂ only through the residuals. If m̂ is slightly wrong, s̃ is slightly off — but the
          moment condition was constructed so that small shifts in s̃ change the numerator and denominator in a way
          that cancels in θ, to first order. The damage from nuisance-model error is <em>second-order</em> — roughly,
          the error squared. A nuisance model that's 90% right contaminates θ by ~1% worth of bias, not 10%.
        </p>
        <Analogy>
          A seesaw balanced exactly at its center: pushing down on one side and up on the other by the same amount
          changes the tilt not at all to first order. The moment condition is the balance point; the nuisance
          predictions are the two hands. Sloppy hands at the wrong spot tip the seesaw; hands near the balance point
          barely matter. This is the precise sense in which "ML does the hard part but doesn't ruin the answer".
        </Analogy>
        <p>
          Two honest caveats. First-order robustness ≠ immunity: a badly wrong nuisance model still biases θ, just
          less than intuition fears. And orthogonality buys you robustness to <em>estimation error</em> in the
          nuisances, not to <em>omitted variables</em> — leave something important out of X and no amount of
          cross-fitting repairs it.
        </p>
      </Section>

      <Section id="nuisance" title="8 · What 'good nuisances' look like">
        <p>
          Because of orthogonality, the nuisance models don't need to be great — they need to be evaluated honestly
          (out-of-fold) and better than nothing. The app reports two out-of-sample R² values for exactly this:
        </p>
        <ul>
          <li><code>r2_y</code> — how well the controls predict the equity move. In our runs it's <em>negative</em>:
            the models predict next-meeting equity moves worse than the historical average. Don't be alarmed — daily
            equity moves are close to unpredictable by design (efficient markets), so a negative out-of-fold R² is
            the honest answer, and a positive one would smell like leakage.</li>
          <li><code>r2_s</code> — how well the controls predict the surprise. This is the diagnostic that matters
            most. In the primary run it's ≈ 0 (−0.003): surprises are essentially unpredictable from pre-announcement
            state. That's the comfortable regime: s̃ ≈ s, the denominator is large, and θ is well-identified. If r2_s
            were high and positive, surprises would be largely "expected given X", residual variation would be thin,
            and θ would be a fragile ratio of small numbers.</li>
        </ul>
        <p>
          The other side of the coin is <code>sd_s_resid</code> — the standard deviation of the residual surprise.
          It is the scale of the denominator. Small values mean few effective "doses" were administered: the Fed
          rarely deviated from what the backdrop predicted, so identifying the dose-response relationship is hard,
          whatever the estimator.
        </p>
        <Analogy>
          r2_s ≈ 0 means every meeting administered a full-strength, unforecastable dose — like a clinical trial
          where every patient actually took a random dose. r2_s near 1 would mean doses were fully predicted by
          patient characteristics — you could only compare patients who took similar doses, and the trial would say
          little.
        </Analogy>
        <p>
          Which learner to pick? Ridge (default) is linear, fast, stable. Elastic net prunes correlated controls.
          Random forests capture nonlinearity and interactions but cost a minute per fit and can overfit if tuned
          greedily — watch whether θ moves when you switch, and whether r2_y/r2_s move with it. A θ that's stable
          across nuisance choices is evidence the functional form of g and m doesn't drive your conclusion.
        </p>
      </Section>

      <Section id="inference" title="9 · Getting honest error bars">
        <p>
          A point estimate without a standard error is a rumor. DML's inference comes from the per-meeting{" "}
          <strong>score</strong> contributions:
        </p>
        <F>ψ<sub>t</sub> = s̃<sub>t</sub> · (ỹ<sub>t</sub> − θ·s̃<sub>t</sub>)</F>
        <p>
          — how much each meeting, weighted by its surprise content, disagrees with the estimated θ. The standard
          error is built from the variance of these scores across meetings. That much is classical. The project's
          two twists are about <em>dependence</em>:
        </p>
        <ul>
          <li><strong>HC1 vs HAC</strong>. Classical DML theory assumes independent observations; ours are
            time-ordered. Adjacent meetings' scores can be correlated (the same policy episode, the same market
            regime). HC1 assumes no correlation; HAC (heteroskedasticity- and autocorrelation-consistent) adds a
            Bartlett-weighted sum of lagged score covariances over meeting order, with <code>hac_lag</code> choosing
            how many meetings back to look. If the HAC interval is noticeably wider than HC1, serial correlation was
            real and HC1 was overconfident.</li>
          <li><strong>Whole-meeting bootstrap</strong>. The nonparametric check: resample <em>meetings</em> with
            replacement, recompute θ on each resample, and read the interval off the distribution. Resampling rows
            would pretend each row is an independent draw; meetings are the independent units here (each row <em>is</em>{" "}
            a meeting in Q1, but the discipline matters for the panel questions to come — and for keeping the unit of
            analysis honest). <code>n_boot</code> controls the number of draws; the app compares this percentile
            interval against the analytic ones. Three intervals that roughly agree = confidence in the confidence
            intervals.</li>
        </ul>
        <Analogy>
          Bootstrapping meetings vs rows: photocopying entire patient charts vs photocopying individual blood tests.
          If a patient's tests are internally correlated (they are), shuffling tests shuffles away the very structure
          you're trying to account for.
        </Analogy>
        <p>
          One more honesty note, stated in the app itself: the bootstrap resamples the scores <em>without refitting
          the nuisance models</em> per draw. That's the cheap bootstrap — it captures sampling uncertainty in θ
          given the nuisances, not uncertainty about the nuisances. It's the standard practice, but it's why we
          cross-fit and check stability across nuisance learners rather than relying on any single interval.
        </p>
      </Section>

      <Section id="ui" title="10 · Every knob in the Playground, mapped to the math">
        <table className="table">
          <thead><tr><th>knob</th><th>what it is in the math above</th></tr></thead>
          <tbody>
            <tr><td>treatment (STMT/MP1/ME)</td><td>which surprise series plays the role of s</td></tr>
            <tr><td>outcome (usmpd_sp500 / etf_day0)</td><td>which series plays Y, and which X set comes with it</td></tr>
            <tr><td>controls checkboxes</td><td>the X inside g(X) and m(X) — leave-one-out for each confounder candidate</td></tr>
            <tr><td>nuisance (+ alpha)</td><td>the learner and regularization used for l̂ = E[Y|X], m̂ = E[s|X]</td></tr>
            <tr><td>min_train / test_size / step</td><td>geometry of the walk-forward cross-fit; min_train meetings are trained-on-only</td></tr>
            <tr><td>embargo</td><td>extra meetings dropped between train and test (for overlapping outcome windows)</td></tr>
            <tr><td>se / hac_lag</td><td>HC1 vs Bartlett-HAC on the score sequence over meeting order</td></tr>
            <tr><td>n_boot</td><td>draws of the whole-meeting bootstrap (0 disables)</td></tr>
            <tr><td>theta, CI</td><td>θ = Σs̃ỹ/Σs̃² and its interval from the score</td></tr>
            <tr><td>residualized fit chart</td><td>ỹ vs s̃; the line through the origin has slope θ — literally the formula, drawn</td></tr>
            <tr><td>influence chart</td><td>θ recomputed dropping one meeting at a time — who is carrying your estimate?</td></tr>
          </tbody>
        </table>
      </Section>

      <Section id="diagnostics" title="11 · Diagnostics and failure modes">
        <p>How to know whether to believe a number the Playground prints. Run these checks in order:</p>
        <ul>
          <li><strong>Sign and size stability.</strong> Flip nuisance ridge → rf, and treatment across STMT/MP1/ME.
            Our saved runs: θ ≈ −0.96 (STMT, SP500 window) and ≈ −0.56 (MP1) — different treatments legitimately
            differ, but one configuration wildly out of family is a red flag.</li>
          <li><strong>Influence.</strong> The LOO chart's top bar is the meeting whose removal moves θ most. On the
            primary configuration it is 2007-09-18, the intermeeting cut — an enormous surprise in a panicked market.
            One meeting dominating the estimate means the answer is partly a story about that meeting; report it that
            way.</li>
          <li><strong>r2_s and sd_s_resid.</strong> Near-zero r2_s with healthy sd_s_resid: comfortable. High r2_s or
            tiny sd_s_resid: the treatment is largely predictable or barely varies — thin denominator, fragile θ.</li>
          <li><strong>Interval agreement.</strong> HC1 vs HAC vs bootstrap. Wide disagreement says the iid assumption
            was doing silent work; trust the widest.</li>
          <li><strong>Discarded share.</strong> n_meetings_discarded / n_meetings_total. Defaults discard ~half the
            sample — acceptable, but a small <code>min_train</code> with unstable θ across fold geometry means you're
            reading noise from too-short training windows.</li>
          <li><strong>Zero outcomes.</strong> The audit lists zero-return meetings (halts, holidays glued to
            meetings). A cluster of them can drag θ mechanically.</li>
        </ul>
        <Warning>
          What DML cannot fix: controls that are measured <em>after</em> the treatment (post-treatment bias —
          blocked by construction here, but the moment you add your own data, it's on you), omitted confounders,
          and a misspecified linear-in-s structure when the truth is strongly nonlinear in the dose. The method is
          honest machinery around assumptions; it doesn't replace them.
        </Warning>
        <p>
          <strong>References.</strong> Chernozhukov, Chetverikov, Demirer et al., "Double/Debiased Machine Learning
          for Treatment and Structural Parameters", <em>The Econometrics Journal</em> (2018) — the modern statement.
          Robinson, "Root-N-Consistent Semiparametric Regression", <em>Econometrica</em> (1988) — the partially
          linear model and partialling-out. Frisch–Waugh–Lovell theorem — why residualized regression works. Project
          design notes: <code>docs/methods.md</code> and <code>docs/question.md</code>.
        </p>
      </Section>
    </div>
  );
}
