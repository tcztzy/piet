from itertools import permutations

MAX_BASES = 200
MIN_BASES = 100

CRUMB = set(range(2**2))

GRAPHS = tuple(
    sorted(
        [
            list(r0),
            [*r1, r0[2]],
            [*r2, r1[1], r0[1]],
            [r3, r2[0], r1[0], r0[0]],
        ]
        for r0 in permutations(CRUMB)
        for r1 in permutations(CRUMB - {r0[2]})
        if r0[0] != r1[0] and r0[1] != r1[1]
        for r2 in permutations(CRUMB - {r0[1], r1[1]})
        if r2[0] not in [r0[0], r1[0]] and r2[1] not in [r0[1], r1[1]]
        for r3 in (CRUMB - {r0[0], r1[0], r2[0]})
    )
)
