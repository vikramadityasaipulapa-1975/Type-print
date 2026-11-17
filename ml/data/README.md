# Timing sample dataset contract

Store one JSON object per line in a local JSONL file. This contract intentionally excludes the typed passage and arbitrary key text.

```json
{"sample_id":"sample-001","session_id":"session-001","user_id":"subject-a","events":[{"id":0,"code":"KeyT","downMs":10.2,"upMs":81.4,"dwellMs":71.2,"flightMs":null},{"id":1,"code":"KeyH","downMs":102.0,"upMs":158.0,"dwellMs":56.0,"flightMs":20.6}]}
```

`code` must be a browser physical key category such as `KeyT`, `Digit2`, `Space`, `Period`, or `Backspace`; never put `KeyboardEvent.key` or the actual passage here. Each `session_id` represents one collection session and is the grouping key for train/validation/test splitting. `user_id` is a modeling label, not an account credential. The JSONL file is not generated from fake/demo user records. Samples should only be added with explicit consent and appropriate protection because timing data is biometric information.

