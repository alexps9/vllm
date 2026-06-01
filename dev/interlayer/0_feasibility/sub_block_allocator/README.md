# sub_block_allocator — feasibility check (phase 0)

two-level allocator (page <-> sub-blocks) is memory-safe under fuzz. Pass(ideal): >=1e6 randomized+adversarial ops, ZERO invariant violations.

Full spec + ideal pass bar: [`../../design.md`](../../design.md) verification gate.
