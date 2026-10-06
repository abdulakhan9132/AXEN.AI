import json, re, random, zlib
from pathlib import Path
import torch
import torch.nn as nn

DIM = 4096
SEED = {
    "chat":       ["hi", "how are you", "tell me a joke", "who are you", "what can you do"],
    "code":       ["write a python function to sort a list", "fix this bug in my react component",
                   "explain this javascript error", "create a fastapi endpoint", "refactor this class"],
    "math":       ["solve x^2 + 3x - 4 = 0", "integrate sin(x)", "what is 15% of 2300",
                   "differentiate x^3 * e^x", "simplify this expression"],
    "file_op":    ["read the file config.json", "list files in my downloads folder",
                   "create a new folder called reports", "delete old logs", "open main.py"],
    "web_search": ["search the web for latest python release", "what is the news today",
                   "find the price of rtx 4060", "look up this company online", "google this for me"],
    "ui_design":  ["design a login screen", "make a dashboard layout", "create a landing page mockup",
                   "suggest a color palette for my app", "build a figma style card component"],
    "memory":     ["remember that my name is abdullah", "what did i tell you yesterday",
                   "save this note", "recall my last project", "forget that"],
    "job_search": ["find ui ux designer jobs in bengaluru", "match my cv to this job",
                   "write a cover letter", "track my applications", "improve my resume"],
    "image_gen":  ["generate an image of a sunset over mountains", "draw a logo for my brand",
                   "create a poster for a music festival", "make a picture of a cyberpunk city",
                   "generate app icon art"],
}

def featurize(text: str) -> torch.Tensor:
    t = text.lower().strip()
    words = re.findall(r"\w+", t)
    feats = ["w:" + w for w in words]
    feats += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
    p = f" {t} "
    for n in (3, 4):
        feats += ["c:" + p[i:i + n] for i in range(len(p) - n + 1)]
    v = torch.zeros(DIM)
    for f in feats:
        v[zlib.crc32(f.encode()) % DIM] += 1.0   # crc32 = stable across runs
    v = torch.log1p(v)
    return v / (v.norm() + 1e-8)

class Net(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_out))
    def forward(self, x): return self.f(x)

class AxenBrain:
    def __init__(self, path="axen_brain.pt", threshold=0.55):
        self.path, self.threshold = Path(path), threshold
        self.labels, self.net = [], None
        if self.path.exists():
            self.load()

    # data: list of (text, intent)
    def train(self, data, epochs=60, lr=2e-3):
        self.labels = sorted({y for _, y in data})
        idx = {l: i for i, l in enumerate(self.labels)}
        random.shuffle(data)
        cut = max(1, int(len(data) * 0.85))
        tr, va = data[:cut], data[cut:] or data[:cut]
        X = torch.stack([featurize(t) for t, _ in tr]); Y = torch.tensor([idx[y] for _, y in tr])
        Xv = torch.stack([featurize(t) for t, _ in va]); Yv = torch.tensor([idx[y] for _, y in va])
        self.net = Net(len(self.labels))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 32):
                b = perm[i:i + 32]
                opt.zero_grad(); lossf(self.net(X[b]), Y[b]).backward(); opt.step()
            if ep % 10 == 9 or ep == epochs - 1:
                self.net.eval()
                with torch.no_grad():
                    acc = (self.net(Xv).argmax(1) == Yv).float().mean().item()
                print(f"epoch {ep+1}: val accuracy {acc:.2%}")
        self.save()

    @torch.no_grad()
    def predict(self, text):
        if self.net is None:
            return {"intent": "unknown", "confidence": 0.0}
        self.net.eval()
        p = torch.softmax(self.net(featurize(text).unsqueeze(0))[0], 0)
        conf, i = p.max(0)
        intent = self.labels[i] if conf >= self.threshold else "unknown"
        return {"intent": intent, "confidence": round(conf.item(), 3)}

    def save(self):
        if self.net is not None:
            torch.save({"labels": self.labels, "state": self.net.state_dict()}, self.path)

    def load(self):
        ck = torch.load(self.path, map_location="cpu")
        self.labels = ck["labels"]
        self.net = Net(len(self.labels))
        self.net.load_state_dict(ck["state"])
        self.net.eval()

def load_data(jsonl="training/intents.jsonl"):
    data = [(t, y) for y, ts in SEED.items() for t in ts]
    p = Path(jsonl)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                data.append((r["text"], r["intent"]))
    return data

if __name__ == "__main__":
    brain = AxenBrain()
    brain.train(load_data())
    for q in ["write me a react navbar", "integrate x^2", "find jobs for product designer",
              "make a pricing page layout", "open notes.txt"]:
        print(q, "->", brain.predict(q))
