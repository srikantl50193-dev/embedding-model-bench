# Tuning misses, read by hand

Fine Print's keyword test can fail a chunk that answers a question in other words. To see whether
this explains Granite's low score, Claude read the top 5 for every tuning question (n=31) each of
three models missed at hit@5. Reproduce the listings with `scripts/inspect_misses.py <model>`.

A question is credited only when a retrieved chunk from the right document visibly states the
answer. Sections that are merely on topic are not credited.

| Model | Keyword hit@5 | Misses read | Answered in other words | Credited hit@5 |
| --- | --- | --- | --- | --- |
| bge-base-en-v1.5 | 23/31 | 8 | 4 | 27/31 |
| bge-small-en-v1.5 | 20/31 | 11 | 5 | 25/31 |
| granite-small-en-r2 | 13/31 | 18 | 3 | 16/31 |

## Questions credited

| Question (shortened) | Required keyword | What the chunk says | Credited to |
| --- | --- | --- | --- |
| Zoom shutting down a paid service | discontinue the services | Zoom may terminate "for any reason or no reason" on 30 business days' notice | base, small, Granite |
| Strava data after closing an account | request account deletion | Up to 45 days to fully delete; some information retained | base, small, Granite |
| Pinterest ads from websites visited | electric guitars | Browsing sleeping bags on a partner's site can lead to ads | base, small, Granite |
| Pinterest's use of uploaded pictures | royalty-free | "we can show it to people and others can use it and save it" | base |
| Spotify group lawsuits | class action | Claims only "in an individual capacity", no class member | small |
| Discord selling data | sell your personal information | "We don't make money from selling your personal information" | small |

## Reading

- The keyword test undercounts every model by a similar amount, 3 to 5 questions. It does not
  explain Granite's gap: Granite is still last, 9 or more questions behind.
- Granite's misses are mostly real. It often reaches the right document but returns its general
  sections, such as contact details and introductions, rather than the passage that answers.
- Some required keywords are brittle. "electric guitars" is a single worked example, and "sell"
  doesn't match the text "selling your personal information" as a phrase.

## Limits

- One reader, who knew which model produced each list, so this is not a blind judgment.
- Only misses were read. Keyword hits were assumed correct.
- Chunks were cut off for reading: 650 characters for Granite, 500 for the BGE models. An answer
  later in a chunk would not have been credited, which could slightly favour Granite here.
- This is a side analysis. All reported scores and the selection rule still use the keyword test.
