# LinkedIn post — the vector index

Upload `vector-index-carousel.pdf` as a **document post**, not as images: it
renders as a native swipeable carousel and reaches further. Regenerate the PDF
with `./render.sh` after editing the HTML.

LinkedIn truncates the body at roughly 210 characters behind a "see more", so
the first two lines carry the click. That is why the hook leads on the
percentage rather than on the technology.

---

## Post copy

**Our AI agent was using 4% of its computer. We gave it 59%, and it made the thing genuinely better.**

`{ "error": "ramMib exceeds the maximum of 16" }`

The agent runs as a BareMetal unikernel — one program, no kernel, no shell, no init, no container, no disk. It boots straight into `main()`.

Its memory ceiling is not a plan you upgrade. It is the platform maximum: 16 MiB. And we were using 640 KB of it, which felt tidy and was actually just waste.

So we spent the rest on a vector database with no database under it. Every note the agent keeps is embedded into a 768-dimension vector held in a static array. Ask it something and it embeds your question, then compares it against every resident vector — one pass, no tree, no index structure, no disk. It now finds "the unikernel starts in 31 ms" when you ask about boot times, which is what you wanted and not what you typed.

Three things we learned:

**Truncated embeddings are not unit vectors.** The model returns 3072 dimensions and truncates to 768 on request. We assumed the result was still unit length. It isn't — measured norm 0.59. Cosine similarity over unnormalised vectors ranks *longer* notes above *closer* ones, silently and plausibly. You would never catch it from the output.

**8 MiB of RAM cost 31 KB of image.** Uninitialised statics are not stored in the binary. The upload went from 2,880,384 to 2,911,872 bytes; the machine just wakes up with the room already claimed.

**The best decision was picking a feature that fails gracefully.** That morning we had measured this platform corrupting 64-bit arithmetic at roughly 1 result in 4000 under load. A brute-force dot product over a big array is *exactly* that loop shape. We built it anyway — because a corrupted score reorders two neighbours in a ranked list. It cannot crash. It cannot invent a memory that was never stored. Anything needing exact arithmetic would have been the wrong thing to put on this machine.

The constraint picked the feature, not the other way round.

Live now: 2,560 slots, 8,180 KB resident, search in 4.2 microseconds, 5 of 5 test queries returning the note a human would pick.

Open source, C, no heap, no OS: github.com/tabibazar/baremetal-agent

*(Built on Return Infinity's BareMetal. The arithmetic bug is reported upstream with a 60-line reproducer.)*

---

## Notes

Put the repository link in the first comment rather than the body if reach
matters more than convenience.

Two limitations are deliberately absent from the copy and present in the
README, because they are worth discussing in the comments rather than
defending: the index is RAM-only, so a restart empties it and the first search
afterwards re-embeds the most recent 32 notes; and float32 was chosen over int8
quantisation, which would have given four times the capacity. Neither is a
gotcha if it is already written down.

## Where every number comes from

| Claim | Source |
|---|---|
| 16 MiB ceiling | `ramMib exceeds the maximum of 16`, returned by the BareMetal Cloud API |
| 640 KB → 9,588 KB | the agent's own startup banner, before and after |
| 59% of RAM | printed by the agent at boot, computed from `RAM_MIB` |
| 2,880,384 → 2,911,872 bytes | `deploy_baremetal.sh` reports the image size each build |
| norm 0.59 | measured against the live embeddings endpoint while designing the feature |
| 1 result in 4000 | [the same-host comparison](https://github.com/tabibazar/unikernel-c/tree/main/docs/nanos-vs-baremetal), five runs, with a Linux control under the same hypervisor |
| 4.2 µs, 5 of 5 | `test_index.c`, run against the live endpoint |
| 31 ms boot | median of five, program start to the application's first printed line |
