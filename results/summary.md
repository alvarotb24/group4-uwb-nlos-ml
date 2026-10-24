# RQ2 latency summary

Sessions: [1, 2, 3]; timed requests: 36000; failed requests: 0.

- TCP connection set-up time to H, median per session: session 1: 0.28 ms, session 2: 0.52 ms, session 3: 0.38 ms
- TCP connection set-up time to C, median per session: session 1: 5.71 ms, session 2: 7.69 ms, session 3: 5.53 ms

## M2 (LR), batch size 1
- Medians: L 0.185 ms, H 2.32 ms, C 6.53 ms.
- Added latency of the cloud service: Delta = 6.34 ms (95 % CI 6.23 to 6.48 ms).
- Split (medians, approximately additive): computation on the VM 0.560 ms (local in-process: 0.185 ms); HTTP/JSON overhead 1.82 ms (from H); network 4.15 ms; network share of Delta about 65 %.
- 99th percentile in C: 16.1 ms; 0.0 % of C requests exceed 100 ms.
- Local REST service vs in-process: 2.14 ms (95 % CI 2.08 to 2.22 ms).

## M4 (RF), batch size 1
- Medians: L 13.1 ms, H 17.9 ms, C 86.6 ms.
- Added latency of the cloud service: Delta = 73.5 ms (95 % CI 73.0 to 73.9 ms).
- Split (medians, approximately additive): computation on the VM 74.0 ms (local in-process: 13.1 ms); HTTP/JSON overhead 2.39 ms (from H); network 6.45 ms; network share of Delta about 9 %.
- 99th percentile in C: 162.8 ms; 13.1 % of C requests exceed 100 ms.
- Local REST service vs in-process: 4.80 ms (95 % CI 4.33 to 5.39 ms).

## M2 (LR), batch size 4
- Medians: L 0.176 ms, H 1.84 ms, C 6.42 ms.
- Added latency of the cloud service: Delta = 6.25 ms (95 % CI 6.11 to 6.38 ms).
- Split (medians, approximately additive): computation on the VM 0.563 ms (local in-process: 0.176 ms); HTTP/JSON overhead 1.53 ms (from H); network 4.33 ms; network share of Delta about 69 %.
- 99th percentile in C: 18.8 ms; 0.2 % of C requests exceed 100 ms.
- Local REST service vs in-process: 1.67 ms (95 % CI 1.65 to 1.69 ms).

## M4 (RF), batch size 4
- Medians: L 15.3 ms, H 14.1 ms, C 88.6 ms.
- Added latency of the cloud service: Delta = 73.2 ms (95 % CI 72.7 to 73.6 ms).
- Split (medians, approximately additive): computation on the VM 75.3 ms (local in-process: 15.3 ms); HTTP/JSON overhead 1.93 ms (from H); network 7.33 ms; network share of Delta about 10 %.
- 99th percentile in C: 181.2 ms; 14.6 % of C requests exceed 100 ms.
- Local REST service vs in-process: -1.202 ms (95 % CI -1.598 to -0.929 ms).

Per-session medians (stability check) are in per_session.csv.
