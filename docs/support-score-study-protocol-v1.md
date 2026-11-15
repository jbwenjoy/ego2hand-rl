# Preregistered offline support-score comparison v1

Freeze config/source before evaluating candidate scores. Read exactly the v1
registered pilot memories and v2 saved actual mean-policy executions, plus the
same365 valid nominal reference targets. Do not fit a policy, select clips,
change physics or change old rewards. Report all four variants on identical
rows with the original force.3N/torque.003Nm scales and old eligibility flags.

Let L(w,t)=mean(((F-Ft)/.3)^2)+mean(((M-Mt)/.003)^2), s=exp(-L/2),
b=exp(-L(0,t)/2), and c=mean(all robot-hand normal loads over8 native samples).
All torques are about moving COM in source BODY axes, exactly as in v1.

* ABSOLUTE: s, the unchanged v1 score.
* CONTACT_ABSOLUTE: 1[c>.1N]s.
* CENTERED: s-b.
* CONTACT_CENTERED: 1[c>.1N](s-b).

Original invalid-target,geometry/native-failure and phase gates disable all
scores. Mean-load threshold is a candidate objective definition, not a new
physical gate or a claim that every native sample is in contact.

Analytic cases include force-only,torque-only and mixed targets; measured wrench
alpha*t for alpha=-1,0,.5,1,1.5,2; weak/zero versus strong balanced squeeze
(zero net wrench with different total normal load); and the two sides of.1N.
For a nonzero target, exact matching should be the unique best alpha in the
grid. A zero target/zero measurement tie is expected and supplies no evidence
of support. Test source-axis rotation invariance and force/torque sensitivity.

Predeclared selection: zero score at zero net wrench for BOTH weak and strong
contact; no artificial threshold discontinuity; exact nonzero target strictly
best; opposite target negative. Select CENTERED only if these invariants hold.
Do not choose by the largest mean score or by apparent training improvement.
Save negative tests against the other variants rather than calling their
expected invariant failures a failed analysis.

Centering can change termination/episode-length incentives because the baseline
depends on reference phase. It is a changed objective, not policy-invariant
potential shaping. A positive centered score means closer to nominal wrench
than zero, not correct tracking/contact feasibility. A balanced squeeze remains
zero; object/self/joint/nonhand geometry gates stay in force. No bound on normal
squeeze is thereby proven. Dense nominal targets remain uncertain simulator
priors. Any selected production learner must carry a new config/schema, pass
GPU/frame/reward/load qualification, and get a newly declared equal budget.
