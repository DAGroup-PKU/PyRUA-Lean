# Authority, scope and authorization

The same task uses one explicit goal from the request, prompt and payload through to review.

**Conflicts.** Within the host platform's higher-level instructions, resolve conflicts in this order: the user's current
explicit request; still-valid user decisions with a matching scope; reproducible raw evidence; summaries; progress notes
and drafts. A later date does not turn an unapproved inference into a user decision. Raw evidence can overturn a
judgment that something works, but it cannot create authorization.

**Exceptions.** Each exception records the rule ID, the user's exact words or a verifiable source, the task/dataset/run,
the axes allowed to change, the time and the expiry conditions. A note that OOD was allowed, or a file name containing
ood, is not enough. Provider choice, GPU count, paid quota and data target are separate dimensions.

**Mode and references.** If the mode is unclear, ask first. Both OOD and non-OOD use real-world images and reach
real-world style. When real references are unavailable, pause rendering and state what is missing; do not switch to
text-only or structure-image generation. Reference leakage is a quality problem to fix and gives no permission to drop
the real references.

**Granted work.** Do not ask again for work the user already authorized, including GPU use, paid-provider calls and
recovery within the granted scope. Anything beyond the scope, or any other service, needs its own authorization. Asking
about the mode obtains goal information; it does not re-approve compute already granted. GPU and paid-provider
authorization comes from the current deployment and session; obtaining the repository grants none.

**One source of rules.** Long-lived behavior rules live in this repository. External AGENTS files and skills may only
link to the canonical version or distribute copies of the same version; they must not keep a semantically independent
set of rules.

**Technical decisions.** The orchestrating agent generates and measures the technical choices of a task, including the
adapter, judge, scene, brief and prompt. Do not turn these outputs into a to-do for the user. A flag-for-human from the
expert author goes to the orchestrator first; the name does not mean every technical failure interrupts the user.
