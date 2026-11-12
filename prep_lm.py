import os, torch
N = 20_000_000
a = open("text8", "rb").read()[:N]
torch.save(torch.frombuffer(bytearray(a), dtype=torch.uint8), "dataA.pt")
lib = os.path.dirname(os.__file__)
parts, tot = [], 0
for root, _, files in os.walk(lib):
    if "site-packages" in root or "test" in root:
        continue
    for f in sorted(files):
        if f.endswith(".py"):
            try:
                b = open(os.path.join(root, f), "rb").read()
            except OSError:
                continue
            if len(b) < 300_000:
                parts.append(b); tot += len(b)
        if tot >= N:
            break
    if tot >= N:
        break
bb = b"\n".join(parts)[:N]
torch.save(torch.frombuffer(bytearray(bb), dtype=torch.uint8), "dataB.pt")
print("A: %d octets | B: %d octets (source: %s)" % (len(a), len(bb), lib))
print("apercu A:", a[:80])
print("apercu B:", bb[:80])
